#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_common.cuh"
#include "gaussian_basic.cuh"
#include "gaussian_enhanced.cuh"
#include "gpu_image.cuh"
#include "laplacian_basic.cuh"
#include "laplacian_enhanced.cuh"
#include "laplacian_threshold_fused.cuh"
#include "median_basic.cuh"
#include "median_enhanced.cuh"
#include "pipeline_basic.cuh"
#include "pipeline_async.cuh"
#include "pipeline_cuda_graph.cuh"
#include "pipeline_cuda_graph_enhanced.cuh"
#include "pipeline_experimental.cuh"
#include "smoke_test.hpp"
#include "sobel_basic.cuh"
#include "sobel_enhanced.cuh"
#include "threshold_basic.cuh"
#include "threshold_enhanced.cuh"

#include <cstring>

namespace py = pybind11;
using namespace xray_cuda;

namespace {

// Trivial pure-C++ function used to verify the Python -> C++ -> compiled
// extension path independently of CUDA.
int add_ints(int a, int b) { return a + b; }

bool cuda_available() { return cuda_is_available(); }

py::dict device_info() {
    py::dict result;
    if (!cuda_is_available()) {
        result["available"] = false;
        result["error"] = "No usable CUDA device found on this system.";
        return result;
    }

    DeviceInfo info = get_device_info();
    result["available"] = true;
    result["name"] = info.name;
    result["compute_capability"] =
        std::to_string(info.compute_capability_major) + "." +
        std::to_string(info.compute_capability_minor);
    result["compute_capability_major"] = info.compute_capability_major;
    result["compute_capability_minor"] = info.compute_capability_minor;
    result["total_global_mem_bytes"] = info.total_global_mem_bytes;
    result["shared_mem_per_block_bytes"] = info.shared_mem_per_block_bytes;
    result["registers_per_block"] = info.registers_per_block;
    result["max_threads_per_block"] = info.max_threads_per_block;
    result["warp_size"] = info.warp_size;
    result["multiprocessor_count"] = info.multiprocessor_count;

    // CUDA version int encoding is major*1000 + minor*10 (e.g. 12060 -> "12.6").
    auto format_cuda_version = [](int encoded) {
        return std::to_string(encoded / 1000) + "." + std::to_string((encoded % 1000) / 10);
    };
    result["driver_version"] = format_cuda_version(info.driver_version);
    result["runtime_version"] = format_cuda_version(info.runtime_version);
    return result;
}

// --------------------------------------------------------------------------
// Section 23: read-only kernel-launch-configuration diagnostic. Wraps the
// EXACT same compute_launch_grid()/default_block_dim() functions every
// production kernel launch already uses (gpu_image.cuh) -- this does not
// duplicate that formula in Python (spec item 49's explicit requirement),
// it just exposes the real function's result for UI display. Performs no
// allocation, no kernel launch, no compute -- purely arithmetic over the
// same dim3 values production code computes for a real call.
// --------------------------------------------------------------------------
py::dict compute_launch_config_binding(int width, int height, int batch_size, int block_x, int block_y) {
    if (width <= 0 || height <= 0 || batch_size <= 0) {
        throw std::invalid_argument(
            "compute_launch_config: width/height/batch_size must be positive, got width=" +
            std::to_string(width) + " height=" + std::to_string(height) + " batch_size=" + std::to_string(batch_size));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("compute_launch_config: block_x/block_y must be positive.");
    }
    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);

    py::dict result;
    result["block_x"] = block_x;
    result["block_y"] = block_y;
    result["block_z"] = 1;
    result["grid_x"] = static_cast<long long>(grid2d.x);
    result["grid_y"] = static_cast<long long>(grid2d.y);
    result["grid_z"] = static_cast<long long>(batch_size);
    result["threads_per_block"] = static_cast<long long>(block_x) * block_y;
    result["total_threads_launched"] =
        static_cast<long long>(grid2d.x) * grid2d.y * batch_size * block_x * block_y;
    return result;
}

py::dict smoke_test() {
    if (!cuda_is_available()) {
        throw std::runtime_error("smoke_test() requires a usable CUDA device; none was found.");
    }
    SmokeTestResult result = run_smoke_test();
    py::dict out;
    out["output"] = result.output;
    out["kernel_ms"] = result.kernel_ms;
    return out;
}

// -- Section 4A: image transfer / GPU buffer infrastructure -----------------------

// Accepts a generic py::array (not py::array_t<uint8_t>) so we can reject
// wrong dtype/shape ourselves with a specific, useful message instead of
// relying on pybind11's implicit-cast machinery to silently convert (or
// opaquely fail on) e.g. a float array.
std::shared_ptr<GpuImage> upload_image(py::array image_obj) {
    if (!cuda_is_available()) {
        throw std::runtime_error("upload_image() requires a usable CUDA device; none was found.");
    }
    if (image_obj.ndim() != 2) {
        throw std::invalid_argument(
            "upload_image expects a 2-D grayscale array [H, W]; got ndim=" +
            std::to_string(image_obj.ndim()) + " with shape (" +
            [&] {
                std::string s;
                for (py::ssize_t i = 0; i < image_obj.ndim(); ++i) {
                    if (i) s += ", ";
                    s += std::to_string(image_obj.shape(i));
                }
                return s;
            }() + "). RGB/RGBA arrays and image batches are not supported in this section.");
    }
    if (!image_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "upload_image expects dtype uint8, got " +
            py::str(image_obj.dtype()).cast<std::string>() +
            ". Convert upstream (e.g. via pipeline.image_loader.load_image) rather than "
            "relying on this function to cast.");
    }

    // dtype already confirmed exact; c_style forces a contiguous layout
    // (copying only if the input wasn't already contiguous -- no numeric
    // conversion happens here).
    py::array_t<uint8_t, py::array::c_style> contiguous =
        py::array_t<uint8_t, py::array::c_style>::ensure(image_obj);

    int height = static_cast<int>(contiguous.shape(0));
    int width = static_cast<int>(contiguous.shape(1));
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "upload_image received an empty image: height=" + std::to_string(height) +
            " width=" + std::to_string(width));
    }

    auto gpu_image = std::make_shared<GpuImage>(height, width);
    gpu_image->upload_from_host(contiguous.data(), static_cast<size_t>(height) * static_cast<size_t>(width));
    return gpu_image;
}

py::array_t<uint8_t> download_image(const std::shared_ptr<GpuImage>& gpu_image) {
    if (!gpu_image) {
        throw std::invalid_argument("download_image received a null GpuImage.");
    }
    py::array_t<uint8_t> output({gpu_image->height(), gpu_image->width()});
    gpu_image->download_to_host(
        static_cast<uint8_t*>(output.mutable_data()),
        static_cast<size_t>(gpu_image->height()) * static_cast<size_t>(gpu_image->width()));
    return output;
}

py::dict image_add_one_binding(const std::shared_ptr<GpuImage>& gpu_image) {
    if (!gpu_image) {
        throw std::invalid_argument("image_add_one received a null GpuImage.");
    }
    auto [output, kernel_ms] = image_add_one(*gpu_image);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = kernel_ms;
    return result;
}

// -- Section 4B: Basic CUDA Gaussian blur -----------------------

py::dict gaussian_basic_binding(
    const std::shared_ptr<GpuImage>& gpu_image,
    py::array_t<float, py::array::c_style> coeffs) {
    if (!gpu_image) {
        throw std::invalid_argument("gaussian_basic_gpu received a null GpuImage.");
    }
    if (coeffs.ndim() != 2 || coeffs.shape(0) != coeffs.shape(1)) {
        throw std::invalid_argument(
            "gaussian_basic_gpu expects square coeffs [k, k]; got ndim=" + std::to_string(coeffs.ndim()));
    }
    int kernel_size = static_cast<int>(coeffs.shape(0));

    auto [output, timing] = gaussian_basic(*gpu_image, coeffs.data(), kernel_size);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    return result;
}

// -- Section 6: Enhanced CUDA Gaussian blur -----------------------

py::dict gaussian_enhanced_binding(
    const std::shared_ptr<GpuImage>& gpu_image,
    py::array_t<float, py::array::c_style> coeffs_1d,
    int variant, int block_x, int block_y) {
    if (!gpu_image) {
        throw std::invalid_argument("gaussian_enhanced_gpu received a null GpuImage.");
    }
    if (coeffs_1d.ndim() != 1) {
        throw std::invalid_argument("gaussian_enhanced_gpu expects a 1-D coeffs array [k]; got ndim=" +
                                     std::to_string(coeffs_1d.ndim()));
    }
    if (variant < 0 || variant > static_cast<int>(GaussianVariant::Specialized)) {
        throw std::invalid_argument("gaussian_enhanced_gpu: invalid variant " + std::to_string(variant));
    }
    int kernel_size = static_cast<int>(coeffs_1d.shape(0));

    auto [output, timing] = gaussian_enhanced(
        *gpu_image, coeffs_1d.data(), kernel_size, static_cast<GaussianVariant>(variant), block_x, block_y);

    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    result["horizontal_ms"] = timing.horizontal_ms;
    result["vertical_ms"] = timing.vertical_ms;
    result["kernel_ms"] = timing.kernel_ms();
    return result;
}

py::dict gaussian_enhanced_batch_binding(
    py::array batch_obj,
    py::array_t<float, py::array::c_style> coeffs_1d,
    int variant, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("gaussian_enhanced_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "gaussian_enhanced_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "gaussian_enhanced_batch_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (coeffs_1d.ndim() != 1) {
        throw std::invalid_argument("gaussian_enhanced_batch_gpu expects a 1-D coeffs array [k].");
    }
    if (variant < 0 || variant > static_cast<int>(GaussianVariant::Specialized)) {
        throw std::invalid_argument("gaussian_enhanced_batch_gpu: invalid variant " + std::to_string(variant));
    }

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    int kernel_size = static_cast<int>(coeffs_1d.shape(0));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());

    auto [output, timing] = gaussian_enhanced_batch(
        input, coeffs_1d.data(), kernel_size, static_cast<GaussianVariant>(variant), block_x, block_y);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::vector<uint8_t> host_output(output->nbytes());
    output->download_to_host(host_output.data(), host_output.size());
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    result["horizontal_ms"] = timing.horizontal_ms;
    result["vertical_ms"] = timing.vertical_ms;
    result["kernel_ms"] = timing.kernel_ms();
    return result;
}

// -- Section 4C: Basic CUDA median filter -----------------------

py::dict median_basic_binding(const std::shared_ptr<GpuImage>& gpu_image, int kernel_size) {
    if (!gpu_image) {
        throw std::invalid_argument("median_basic_gpu received a null GpuImage.");
    }
    auto [output, timing] = median_basic(*gpu_image, kernel_size);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 7: Enhanced CUDA median filter -----------------------

py::dict median_enhanced_binding(
    const std::shared_ptr<GpuImage>& gpu_image, int kernel_size, int variant, int block_x, int block_y) {
    if (!gpu_image) {
        throw std::invalid_argument("median_enhanced_gpu received a null GpuImage.");
    }
    if (variant < 0 || variant > static_cast<int>(MedianVariant::Specialized)) {
        throw std::invalid_argument("median_enhanced_gpu: invalid variant " + std::to_string(variant));
    }
    auto [output, timing] = median_enhanced(
        *gpu_image, kernel_size, static_cast<MedianVariant>(variant), block_x, block_y);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

py::dict median_enhanced_batch_binding(
    py::array batch_obj, int kernel_size, int variant, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("median_enhanced_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "median_enhanced_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "median_enhanced_batch_gpu expects dtype uint8, got " + py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (variant < 0 || variant > static_cast<int>(MedianVariant::Specialized)) {
        throw std::invalid_argument("median_enhanced_batch_gpu: invalid variant " + std::to_string(variant));
    }

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());

    auto [output, timing] = median_enhanced_batch(
        input, kernel_size, static_cast<MedianVariant>(variant), block_x, block_y);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::vector<uint8_t> host_output(output->nbytes());
    output->download_to_host(host_output.data(), host_output.size());
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 4D: Basic CUDA Sobel edge detection -----------------------

py::dict sobel_basic_binding(const std::shared_ptr<GpuImage>& gpu_image, int mode) {
    if (!gpu_image) {
        throw std::invalid_argument("sobel_basic_gpu received a null GpuImage.");
    }
    auto [output, timing] = sobel_basic(*gpu_image, mode);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 8: Enhanced CUDA Sobel edge detection -----------------------

py::dict sobel_enhanced_binding(
    const std::shared_ptr<GpuImage>& gpu_image, int mode, int variant, int block_x, int block_y) {
    if (!gpu_image) {
        throw std::invalid_argument("sobel_enhanced_gpu received a null GpuImage.");
    }
    if (variant < 0 || variant > static_cast<int>(SobelVariant::Separable)) {
        throw std::invalid_argument("sobel_enhanced_gpu: invalid variant " + std::to_string(variant));
    }
    auto [output, timing] = sobel_enhanced(
        *gpu_image, mode, static_cast<SobelVariant>(variant), block_x, block_y);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms();
    return result;
}

py::dict sobel_enhanced_batch_binding(
    py::array batch_obj, int mode, int variant, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("sobel_enhanced_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "sobel_enhanced_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "sobel_enhanced_batch_gpu expects dtype uint8, got " + py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (variant < 0 || variant > static_cast<int>(SobelVariant::Separable)) {
        throw std::invalid_argument("sobel_enhanced_batch_gpu: invalid variant " + std::to_string(variant));
    }

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());

    auto [output, timing] = sobel_enhanced_batch(
        input, mode, static_cast<SobelVariant>(variant), block_x, block_y);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::vector<uint8_t> host_output(output->nbytes());
    output->download_to_host(host_output.data(), host_output.size());
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["kernel_ms"] = timing.kernel_ms();
    return result;
}

// -- Section 4E: Basic CUDA Laplacian filter -----------------------

py::dict laplacian_basic_binding(
    const std::shared_ptr<GpuImage>& gpu_image,
    py::array_t<float, py::array::c_style> coeffs,
    float scale, float delta) {
    if (!gpu_image) {
        throw std::invalid_argument("laplacian_basic_gpu received a null GpuImage.");
    }
    if (coeffs.ndim() != 2 || coeffs.shape(0) != coeffs.shape(1)) {
        throw std::invalid_argument(
            "laplacian_basic_gpu expects square coeffs [k, k]; got ndim=" + std::to_string(coeffs.ndim()));
    }
    int kernel_size = static_cast<int>(coeffs.shape(0));

    auto [output, timing] = laplacian_basic(*gpu_image, coeffs.data(), kernel_size, scale, delta);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    return result;
}

// -- Section 9: Enhanced CUDA Laplacian filter -----------------------

py::dict laplacian_enhanced_binding(
    const std::shared_ptr<GpuImage>& gpu_image,
    py::array_t<float, py::array::c_style> coeffs,
    float scale, float delta, int variant, int block_x, int block_y) {
    if (!gpu_image) {
        throw std::invalid_argument("laplacian_enhanced_gpu received a null GpuImage.");
    }
    if (coeffs.ndim() != 2 || coeffs.shape(0) != coeffs.shape(1)) {
        throw std::invalid_argument(
            "laplacian_enhanced_gpu expects square coeffs [k, k]; got ndim=" + std::to_string(coeffs.ndim()));
    }
    if (variant < 0 || variant > static_cast<int>(LaplacianVariant::Explicit)) {
        throw std::invalid_argument("laplacian_enhanced_gpu: invalid variant " + std::to_string(variant));
    }
    int kernel_size = static_cast<int>(coeffs.shape(0));

    auto [output, timing] = laplacian_enhanced(
        *gpu_image, coeffs.data(), kernel_size, scale, delta,
        static_cast<LaplacianVariant>(variant), block_x, block_y);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    return result;
}

py::dict laplacian_enhanced_batch_binding(
    py::array batch_obj, py::array_t<float, py::array::c_style> coeffs,
    float scale, float delta, int variant, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("laplacian_enhanced_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "laplacian_enhanced_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "laplacian_enhanced_batch_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (coeffs.ndim() != 2 || coeffs.shape(0) != coeffs.shape(1)) {
        throw std::invalid_argument("laplacian_enhanced_batch_gpu expects square coeffs [k, k]");
    }
    if (variant < 0 || variant > static_cast<int>(LaplacianVariant::Explicit)) {
        throw std::invalid_argument("laplacian_enhanced_batch_gpu: invalid variant " + std::to_string(variant));
    }
    int kernel_size = static_cast<int>(coeffs.shape(0));

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());

    auto [output, timing] = laplacian_enhanced_batch(
        input, coeffs.data(), kernel_size, scale, delta,
        static_cast<LaplacianVariant>(variant), block_x, block_y);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::vector<uint8_t> host_output(output->nbytes());
    output->download_to_host(host_output.data(), host_output.size());
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["kernel_ms"] = timing.kernel_ms;
    result["coeff_upload_ms"] = timing.coeff_upload_ms;
    return result;
}

// -- Section 4F: Basic CUDA binary threshold -----------------------

py::dict threshold_basic_binding(const std::shared_ptr<GpuImage>& gpu_image, int threshold_value, int max_value) {
    if (!gpu_image) {
        throw std::invalid_argument("threshold_basic_gpu received a null GpuImage.");
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (max_value < 0 || max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(max_value));
    }

    auto [output, timing] = threshold_basic(
        *gpu_image, static_cast<uint8_t>(threshold_value), static_cast<uint8_t>(max_value));
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 10: Enhanced CUDA binary threshold -----------------------

py::dict threshold_enhanced_binding(
    const std::shared_ptr<GpuImage>& gpu_image, int threshold_value, int max_value,
    int variant, int block_x, int block_y) {
    if (!gpu_image) {
        throw std::invalid_argument("threshold_enhanced_gpu received a null GpuImage.");
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (max_value < 0 || max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(max_value));
    }
    if (variant < 0 || variant > static_cast<int>(ThresholdVariant::MultiPixel)) {
        throw std::invalid_argument("threshold_enhanced_gpu: invalid variant " + std::to_string(variant));
    }

    auto [output, timing] = threshold_enhanced(
        *gpu_image, static_cast<uint8_t>(threshold_value), static_cast<uint8_t>(max_value),
        static_cast<ThresholdVariant>(variant), block_x, block_y);
    py::dict result;
    result["output"] = std::shared_ptr<GpuImage>(std::move(output));
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

py::dict threshold_enhanced_batch_binding(
    py::array batch_obj, int threshold_value, int max_value,
    int variant, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("threshold_enhanced_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "threshold_enhanced_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "threshold_enhanced_batch_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (max_value < 0 || max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(max_value));
    }
    if (variant < 0 || variant > static_cast<int>(ThresholdVariant::MultiPixel)) {
        throw std::invalid_argument("threshold_enhanced_batch_gpu: invalid variant " + std::to_string(variant));
    }

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());

    auto [output, timing] = threshold_enhanced_batch(
        input, static_cast<uint8_t>(threshold_value), static_cast<uint8_t>(max_value),
        static_cast<ThresholdVariant>(variant), block_x, block_y);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::vector<uint8_t> host_output(output->nbytes());
    output->download_to_host(host_output.data(), host_output.size());
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 10: experimental fused Laplacian+Threshold kernel -----------------------

py::dict laplacian_threshold_fused_batch_binding(
    py::array batch_obj, py::array_t<float, py::array::c_style> coeffs,
    float scale, float delta, int threshold_value, int max_value, int block_x, int block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error(
            "laplacian_threshold_fused_batch_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "laplacian_threshold_fused_batch_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "laplacian_threshold_fused_batch_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }
    if (coeffs.ndim() != 2 || coeffs.shape(0) != coeffs.shape(1)) {
        throw std::invalid_argument("laplacian_threshold_fused_batch_gpu expects square coeffs [k, k]");
    }
    if (threshold_value < 0 || threshold_value > 255 || max_value < 0 || max_value > 255) {
        throw std::invalid_argument("threshold_value/max_value must be in [0, 255]");
    }
    int kernel_size = static_cast<int>(coeffs.shape(0));

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);
    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));

    GpuImageBatch input(batch_size, height, width);
    input.upload_from_host(batch.data(), batch.size());
    GpuImageBatch output(batch_size, height, width);

    FusedLaplacianThresholdTiming timing{};
    laplacian_threshold_fused_dispatch(
        input.data(), output.data(), batch_size, height, width,
        coeffs.data(), kernel_size, scale, delta,
        static_cast<uint8_t>(threshold_value), static_cast<uint8_t>(max_value),
        block_x, block_y, timing);

    std::vector<uint8_t> host_output(output.nbytes());
    output.download_to_host(host_output.data(), host_output.size());
    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), host_output.data(), host_output.size());

    py::dict result;
    result["output"] = output_array;
    result["kernel_ms"] = timing.kernel_ms;
    return result;
}

// -- Section 5: production Basic CUDA pipeline (single native call) -----------------------

py::object optional_ms(float value) {
    // BasicPipelineTiming uses -1.0f to mean "stage skipped" (see
    // pipeline_basic.cuh); translate that to Python None here so a
    // caller never mistakes "skipped" for "ran in 0ms" -- same
    // None-for-disabled convention as cpu.pipeline.TimingResult.
    if (value < 0.0f) {
        return py::none();
    }
    return py::cast(value);
}

py::dict run_basic_cuda_pipeline_binding(
    py::array batch_obj,
    bool gaussian_enabled, py::array_t<float, py::array::c_style> gaussian_coeffs,
    bool median_enabled, int median_kernel_size,
    bool sobel_enabled, int sobel_mode,
    bool laplacian_enabled, py::array_t<float, py::array::c_style> laplacian_coeffs,
    float laplacian_scale, float laplacian_delta,
    bool threshold_enabled, int threshold_value, int threshold_max_value,
    // Section 6: optional, trailing, defaulted -- existing positional
    // call sites (Section 5's cuda/pipeline.py) are unaffected.
    bool gaussian_use_enhanced, py::object gaussian_coeffs_1d,
    int gaussian_variant, int gaussian_block_x, int gaussian_block_y,
    // Section 7: same pattern, same backward-compatibility guarantee.
    bool median_use_enhanced, int median_variant, int median_block_x, int median_block_y,
    // Section 8: same pattern, same backward-compatibility guarantee.
    bool sobel_use_enhanced, int sobel_variant, int sobel_block_x, int sobel_block_y,
    // Section 9: same pattern, same backward-compatibility guarantee.
    bool laplacian_use_enhanced, int laplacian_variant, int laplacian_block_x, int laplacian_block_y,
    // Section 10: same pattern, same backward-compatibility guarantee.
    bool threshold_use_enhanced, int threshold_variant, int threshold_block_x, int threshold_block_y) {
    if (!cuda_is_available()) {
        throw std::runtime_error("run_basic_cuda_pipeline_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "run_basic_cuda_pipeline_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()) +
            ". A single image must be wrapped as a batch of 1 by the caller.");
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "run_basic_cuda_pipeline_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }

    py::array_t<uint8_t, py::array::c_style> batch =
        py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);

    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "run_basic_cuda_pipeline_gpu received an empty batch: batch_size=" + std::to_string(batch_size) +
            " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (threshold_max_value < 0 || threshold_max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(threshold_max_value));
    }

    BasicPipelineConfig config;
    config.gaussian_enabled = gaussian_enabled;
    py::array_t<float, py::array::c_style> gaussian_coeffs_1d_arr;  // kept alive for the .data() pointer below
    if (gaussian_enabled) {
        config.gaussian_use_enhanced = gaussian_use_enhanced;
        if (gaussian_use_enhanced) {
            if (gaussian_coeffs_1d.is_none()) {
                throw std::invalid_argument(
                    "run_basic_cuda_pipeline_gpu: gaussian_use_enhanced=True requires gaussian_coeffs_1d.");
            }
            gaussian_coeffs_1d_arr = py::array_t<float, py::array::c_style>::ensure(gaussian_coeffs_1d);
            if (!gaussian_coeffs_1d_arr || gaussian_coeffs_1d_arr.ndim() != 1) {
                throw std::invalid_argument("gaussian_coeffs_1d must be a 1-D float array [k].");
            }
            config.gaussian_coeffs_1d = gaussian_coeffs_1d_arr.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs_1d_arr.shape(0));
            if (gaussian_variant < 0 || gaussian_variant > static_cast<int>(GaussianVariant::Specialized)) {
                throw std::invalid_argument("gaussian_variant: invalid value " + std::to_string(gaussian_variant));
            }
            config.gaussian_variant = static_cast<GaussianVariant>(gaussian_variant);
            config.gaussian_block_x = gaussian_block_x;
            config.gaussian_block_y = gaussian_block_y;
        } else {
            if (gaussian_coeffs.ndim() != 2 || gaussian_coeffs.shape(0) != gaussian_coeffs.shape(1)) {
                throw std::invalid_argument("gaussian_coeffs must be a square 2-D array [k, k]");
            }
            config.gaussian_coeffs = gaussian_coeffs.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs.shape(0));
        }
    }

    config.median_enabled = median_enabled;
    config.median_kernel_size = median_kernel_size;
    if (median_enabled && median_use_enhanced) {
        if (median_variant < 0 || median_variant > static_cast<int>(MedianVariant::Specialized)) {
            throw std::invalid_argument("median_variant: invalid value " + std::to_string(median_variant));
        }
        config.median_use_enhanced = true;
        config.median_variant = static_cast<MedianVariant>(median_variant);
        config.median_block_x = median_block_x;
        config.median_block_y = median_block_y;
    }

    config.sobel_enabled = sobel_enabled;
    config.sobel_mode = sobel_mode;
    if (sobel_enabled && sobel_use_enhanced) {
        if (sobel_variant < 0 || sobel_variant > static_cast<int>(SobelVariant::Separable)) {
            throw std::invalid_argument("sobel_variant: invalid value " + std::to_string(sobel_variant));
        }
        config.sobel_use_enhanced = true;
        config.sobel_variant = static_cast<SobelVariant>(sobel_variant);
        config.sobel_block_x = sobel_block_x;
        config.sobel_block_y = sobel_block_y;
    }

    config.laplacian_enabled = laplacian_enabled;
    if (laplacian_enabled) {
        if (laplacian_coeffs.ndim() != 2 || laplacian_coeffs.shape(0) != laplacian_coeffs.shape(1)) {
            throw std::invalid_argument("laplacian_coeffs must be a square 2-D array [k, k]");
        }
        config.laplacian_coeffs = laplacian_coeffs.data();
        config.laplacian_kernel_size = static_cast<int>(laplacian_coeffs.shape(0));
        if (laplacian_use_enhanced) {
            if (laplacian_variant < 0 || laplacian_variant > static_cast<int>(LaplacianVariant::Explicit)) {
                throw std::invalid_argument("laplacian_variant: invalid value " + std::to_string(laplacian_variant));
            }
            config.laplacian_use_enhanced = true;
            config.laplacian_variant = static_cast<LaplacianVariant>(laplacian_variant);
            config.laplacian_block_x = laplacian_block_x;
            config.laplacian_block_y = laplacian_block_y;
        }
    }
    config.laplacian_scale = laplacian_scale;
    config.laplacian_delta = laplacian_delta;

    config.threshold_enabled = threshold_enabled;
    config.threshold_value = static_cast<uint8_t>(threshold_value);
    config.threshold_max_value = static_cast<uint8_t>(threshold_max_value);
    if (threshold_enabled && threshold_use_enhanced) {
        if (threshold_variant < 0 || threshold_variant > static_cast<int>(ThresholdVariant::MultiPixel)) {
            throw std::invalid_argument("threshold_variant: invalid value " + std::to_string(threshold_variant));
        }
        config.threshold_use_enhanced = true;
        config.threshold_variant = static_cast<ThresholdVariant>(threshold_variant);
        config.threshold_block_x = threshold_block_x;
        config.threshold_block_y = threshold_block_y;
    }

    auto [output_bytes, timing] =
        run_basic_cuda_pipeline_batch(batch.data(), batch_size, height, width, config);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), output_bytes.data(), output_bytes.size());

    py::dict result;
    result["output"] = output_array;
    result["h2d_ms"] = timing.h2d_ms;
    result["gaussian_ms"] = optional_ms(timing.gaussian_ms);
    result["median_ms"] = optional_ms(timing.median_ms);
    result["sobel_ms"] = optional_ms(timing.sobel_ms);
    result["laplacian_ms"] = optional_ms(timing.laplacian_ms);
    result["threshold_ms"] = optional_ms(timing.threshold_ms);
    result["d2h_ms"] = timing.d2h_ms;
    result["compute_ms"] = timing.compute_ms();
    result["total_ms"] = timing.total_ms();
    return result;
}

// --------------------------------------------------------------------------
// Section 20B: EXPERIMENTAL ONLY. Mirrors run_basic_cuda_pipeline_binding's
// parameter surface and validation exactly (deliberately NOT refactored
// into a shared helper -- gaussian_coeffs_1d_arr's raw-pointer lifetime
// must stay tied to this function's own scope, matching the production
// binding's own pattern) plus two new memory-strategy flags. Never called
// by cuda/pipeline.py's production run_basic_cuda_pipeline()/
// run_enhanced_cuda_pipeline() -- only by the Section 20B experiment
// harness via its own Python-level wrapper.
// --------------------------------------------------------------------------
py::dict run_experimental_persistent_pinned_binding(
    PersistentCudaPipeline& pipeline,
    py::array batch_obj,
    bool gaussian_enabled, py::array_t<float, py::array::c_style> gaussian_coeffs,
    bool median_enabled, int median_kernel_size,
    bool sobel_enabled, int sobel_mode,
    bool laplacian_enabled, py::array_t<float, py::array::c_style> laplacian_coeffs,
    float laplacian_scale, float laplacian_delta,
    bool threshold_enabled, int threshold_value, int threshold_max_value,
    bool gaussian_use_enhanced, py::object gaussian_coeffs_1d,
    int gaussian_variant, int gaussian_block_x, int gaussian_block_y,
    bool median_use_enhanced, int median_variant, int median_block_x, int median_block_y,
    bool sobel_use_enhanced, int sobel_variant, int sobel_block_x, int sobel_block_y,
    bool laplacian_use_enhanced, int laplacian_variant, int laplacian_block_x, int laplacian_block_y,
    bool threshold_use_enhanced, int threshold_variant, int threshold_block_x, int threshold_block_y,
    bool use_persistent_buffers, bool use_pinned_memory) {
    if (!cuda_is_available()) {
        throw std::runtime_error("run_experimental_persistent_pinned_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "run_experimental_persistent_pinned_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "run_experimental_persistent_pinned_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }

    py::array_t<uint8_t, py::array::c_style> batch = py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);

    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "run_experimental_persistent_pinned_gpu received an empty batch: batch_size=" + std::to_string(batch_size) +
            " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (threshold_max_value < 0 || threshold_max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(threshold_max_value));
    }

    BasicPipelineConfig config;
    config.gaussian_enabled = gaussian_enabled;
    py::array_t<float, py::array::c_style> gaussian_coeffs_1d_arr;  // kept alive for the .data() pointer below
    if (gaussian_enabled) {
        config.gaussian_use_enhanced = gaussian_use_enhanced;
        if (gaussian_use_enhanced) {
            if (gaussian_coeffs_1d.is_none()) {
                throw std::invalid_argument(
                    "run_experimental_persistent_pinned_gpu: gaussian_use_enhanced=True requires gaussian_coeffs_1d.");
            }
            gaussian_coeffs_1d_arr = py::array_t<float, py::array::c_style>::ensure(gaussian_coeffs_1d);
            if (!gaussian_coeffs_1d_arr || gaussian_coeffs_1d_arr.ndim() != 1) {
                throw std::invalid_argument("gaussian_coeffs_1d must be a 1-D float array [k].");
            }
            config.gaussian_coeffs_1d = gaussian_coeffs_1d_arr.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs_1d_arr.shape(0));
            if (gaussian_variant < 0 || gaussian_variant > static_cast<int>(GaussianVariant::Specialized)) {
                throw std::invalid_argument("gaussian_variant: invalid value " + std::to_string(gaussian_variant));
            }
            config.gaussian_variant = static_cast<GaussianVariant>(gaussian_variant);
            config.gaussian_block_x = gaussian_block_x;
            config.gaussian_block_y = gaussian_block_y;
        } else {
            if (gaussian_coeffs.ndim() != 2 || gaussian_coeffs.shape(0) != gaussian_coeffs.shape(1)) {
                throw std::invalid_argument("gaussian_coeffs must be a square 2-D array [k, k]");
            }
            config.gaussian_coeffs = gaussian_coeffs.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs.shape(0));
        }
    }

    config.median_enabled = median_enabled;
    config.median_kernel_size = median_kernel_size;
    if (median_enabled && median_use_enhanced) {
        if (median_variant < 0 || median_variant > static_cast<int>(MedianVariant::Specialized)) {
            throw std::invalid_argument("median_variant: invalid value " + std::to_string(median_variant));
        }
        config.median_use_enhanced = true;
        config.median_variant = static_cast<MedianVariant>(median_variant);
        config.median_block_x = median_block_x;
        config.median_block_y = median_block_y;
    }

    config.sobel_enabled = sobel_enabled;
    config.sobel_mode = sobel_mode;
    if (sobel_enabled && sobel_use_enhanced) {
        if (sobel_variant < 0 || sobel_variant > static_cast<int>(SobelVariant::Separable)) {
            throw std::invalid_argument("sobel_variant: invalid value " + std::to_string(sobel_variant));
        }
        config.sobel_use_enhanced = true;
        config.sobel_variant = static_cast<SobelVariant>(sobel_variant);
        config.sobel_block_x = sobel_block_x;
        config.sobel_block_y = sobel_block_y;
    }

    config.laplacian_enabled = laplacian_enabled;
    if (laplacian_enabled) {
        if (laplacian_coeffs.ndim() != 2 || laplacian_coeffs.shape(0) != laplacian_coeffs.shape(1)) {
            throw std::invalid_argument("laplacian_coeffs must be a square 2-D array [k, k]");
        }
        config.laplacian_coeffs = laplacian_coeffs.data();
        config.laplacian_kernel_size = static_cast<int>(laplacian_coeffs.shape(0));
        if (laplacian_use_enhanced) {
            if (laplacian_variant < 0 || laplacian_variant > static_cast<int>(LaplacianVariant::Explicit)) {
                throw std::invalid_argument("laplacian_variant: invalid value " + std::to_string(laplacian_variant));
            }
            config.laplacian_use_enhanced = true;
            config.laplacian_variant = static_cast<LaplacianVariant>(laplacian_variant);
            config.laplacian_block_x = laplacian_block_x;
            config.laplacian_block_y = laplacian_block_y;
        }
    }
    config.laplacian_scale = laplacian_scale;
    config.laplacian_delta = laplacian_delta;

    config.threshold_enabled = threshold_enabled;
    config.threshold_value = static_cast<uint8_t>(threshold_value);
    config.threshold_max_value = static_cast<uint8_t>(threshold_max_value);
    if (threshold_enabled && threshold_use_enhanced) {
        if (threshold_variant < 0 || threshold_variant > static_cast<int>(ThresholdVariant::MultiPixel)) {
            throw std::invalid_argument("threshold_variant: invalid value " + std::to_string(threshold_variant));
        }
        config.threshold_use_enhanced = true;
        config.threshold_variant = static_cast<ThresholdVariant>(threshold_variant);
        config.threshold_block_x = threshold_block_x;
        config.threshold_block_y = threshold_block_y;
    }

    auto [output_bytes, timing] = pipeline.run(
        batch.data(), batch_size, height, width, config, use_persistent_buffers, use_pinned_memory);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), output_bytes.data(), output_bytes.size());

    py::dict result;
    result["output"] = output_array;
    result["h2d_ms"] = timing.h2d_ms;
    result["gaussian_ms"] = optional_ms(timing.gaussian_ms);
    result["median_ms"] = optional_ms(timing.median_ms);
    result["sobel_ms"] = optional_ms(timing.sobel_ms);
    result["laplacian_ms"] = optional_ms(timing.laplacian_ms);
    result["threshold_ms"] = optional_ms(timing.threshold_ms);
    result["d2h_ms"] = timing.d2h_ms;
    result["compute_ms"] = timing.compute_ms();
    result["total_ms"] = timing.total_ms();
    result["alloc_ms"] = timing.alloc_ms;
    result["host_stage_ms"] = timing.host_stage_ms;
    result["used_persistent_buffers"] = timing.used_persistent_buffers;
    result["used_pinned_memory"] = timing.used_pinned_memory;
    result["grew_gpu_buffers"] = timing.grew_gpu_buffers;
    result["grew_pinned_buffers"] = timing.grew_pinned_buffers;
    return result;
}

// --------------------------------------------------------------------------
// Section 20D: EXPERIMENTAL ONLY. CudaGraphPipeline captures the five
// BASIC (non-Enhanced) __global__ kernels only -- see
// pipeline_cuda_graph.cuh/.cu's file headers for why Enhanced graph
// capture is not offered. Deliberately a much smaller parameter surface
// than run_basic_cuda_pipeline_binding/run_experimental_persistent_pinned_binding
// (no *_use_enhanced/*_variant/*_block_* arguments) since there is exactly
// one kernel implementation per filter to dispatch here.
// --------------------------------------------------------------------------
py::dict run_experimental_cuda_graph_binding(
    CudaGraphPipeline& pipeline,
    py::array batch_obj,
    bool gaussian_enabled, py::array_t<float, py::array::c_style> gaussian_coeffs,
    bool median_enabled, int median_kernel_size,
    bool sobel_enabled, int sobel_mode,
    bool laplacian_enabled, py::array_t<float, py::array::c_style> laplacian_coeffs,
    float laplacian_scale, float laplacian_delta,
    bool threshold_enabled, int threshold_value, int threshold_max_value,
    bool use_graph, bool full_pipeline_scope) {
    if (!cuda_is_available()) {
        throw std::runtime_error("run_experimental_cuda_graph_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "run_experimental_cuda_graph_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "run_experimental_cuda_graph_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }

    py::array_t<uint8_t, py::array::c_style> batch = py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);

    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "run_experimental_cuda_graph_gpu received an empty batch: batch_size=" + std::to_string(batch_size) +
            " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (threshold_max_value < 0 || threshold_max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(threshold_max_value));
    }

    GraphPipelineConfig config;
    config.gaussian_enabled = gaussian_enabled;
    if (gaussian_enabled) {
        if (gaussian_coeffs.ndim() != 2 || gaussian_coeffs.shape(0) != gaussian_coeffs.shape(1)) {
            throw std::invalid_argument("gaussian_coeffs must be a square 2-D array [k, k]");
        }
        config.gaussian_coeffs = gaussian_coeffs.data();
        config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs.shape(0));
        if (config.gaussian_kernel_size <= 0 || config.gaussian_kernel_size % 2 == 0) {
            throw std::invalid_argument(
                "run_experimental_cuda_graph_gpu: invalid gaussian_kernel_size " +
                std::to_string(config.gaussian_kernel_size));
        }
    }

    config.median_enabled = median_enabled;
    config.median_kernel_size = median_kernel_size;
    if (median_enabled && (median_kernel_size <= 0 || median_kernel_size % 2 == 0 ||
                            median_kernel_size > kMedianBasicMaxKernelSize)) {
        throw std::invalid_argument(
            "run_experimental_cuda_graph_gpu: invalid median_kernel_size " + std::to_string(median_kernel_size));
    }

    config.sobel_enabled = sobel_enabled;
    config.sobel_mode = sobel_mode;
    if (sobel_enabled && (sobel_mode < 0 || sobel_mode > static_cast<int>(SobelMode::AbsSum))) {
        throw std::invalid_argument("run_experimental_cuda_graph_gpu: invalid sobel_mode " + std::to_string(sobel_mode));
    }

    config.laplacian_enabled = laplacian_enabled;
    if (laplacian_enabled) {
        if (laplacian_coeffs.ndim() != 2 || laplacian_coeffs.shape(0) != laplacian_coeffs.shape(1)) {
            throw std::invalid_argument("laplacian_coeffs must be a square 2-D array [k, k]");
        }
        config.laplacian_coeffs = laplacian_coeffs.data();
        config.laplacian_kernel_size = static_cast<int>(laplacian_coeffs.shape(0));
        if (config.laplacian_kernel_size <= 0 || config.laplacian_kernel_size % 2 == 0) {
            throw std::invalid_argument(
                "run_experimental_cuda_graph_gpu: invalid laplacian_kernel_size " +
                std::to_string(config.laplacian_kernel_size));
        }
    }
    config.laplacian_scale = laplacian_scale;
    config.laplacian_delta = laplacian_delta;

    config.threshold_enabled = threshold_enabled;
    config.threshold_value = static_cast<uint8_t>(threshold_value);
    config.threshold_max_value = static_cast<uint8_t>(threshold_max_value);

    GraphScope scope = full_pipeline_scope ? GraphScope::FullPipeline : GraphScope::KernelsOnly;
    auto [output_bytes, timing] =
        pipeline.run(batch.data(), batch_size, height, width, config, use_graph, scope);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), output_bytes.data(), output_bytes.size());

    py::dict result;
    result["output"] = output_array;
    result["alloc_ms"] = timing.alloc_ms;
    result["h2d_ms"] = timing.h2d_ms;
    result["compute_ms"] = timing.compute_ms;
    result["d2h_ms"] = timing.d2h_ms;
    result["capture_ms"] = timing.capture_ms;
    result["instantiate_ms"] = timing.instantiate_ms;
    result["node_update_ms"] = timing.node_update_ms;
    result["total_ms"] = timing.total_ms();
    result["used_graph"] = timing.used_graph;
    result["graph_cache_hit"] = timing.graph_cache_hit;
    result["fallback_reason"] = timing.fallback_reason;
    return result;
}

// --------------------------------------------------------------------------
// Section 20E: EXPERIMENTAL ONLY. CudaGraphEnhancedPipeline captures the
// five ENHANCED filter kernels in their production (Specialized/
// Network3x3/Vectorized) configuration only -- see
// pipeline_cuda_graph_enhanced.cuh's file header for why these are
// isolated, verified-identical kernel copies rather than direct calls
// into gaussian_enhanced.cu et al. (those kernels have anonymous-
// namespace/internal linkage). gaussian_coeffs_1d here is the 1-D
// separable array (length kernel_size), NOT the 2-D array the Basic
// binding takes -- matches cuda.gaussian.gaussian_kernel_1d(), same
// convention as cuda/pipeline.py's Enhanced Gaussian call.
// --------------------------------------------------------------------------
py::dict run_experimental_enhanced_cuda_graph_binding(
    CudaGraphEnhancedPipeline& pipeline,
    py::array batch_obj,
    bool gaussian_enabled, py::array_t<float, py::array::c_style> gaussian_coeffs_1d,
    bool median_enabled,
    bool sobel_enabled, int sobel_mode,
    bool laplacian_enabled, py::array_t<float, py::array::c_style> laplacian_coeffs,
    float laplacian_scale, float laplacian_delta,
    bool threshold_enabled, int threshold_value, int threshold_max_value,
    bool use_graph, bool full_pipeline_scope) {
    if (!cuda_is_available()) {
        throw std::runtime_error("run_experimental_enhanced_cuda_graph_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "run_experimental_enhanced_cuda_graph_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "run_experimental_enhanced_cuda_graph_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }

    py::array_t<uint8_t, py::array::c_style> batch = py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);

    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "run_experimental_enhanced_cuda_graph_gpu received an empty batch: batch_size=" + std::to_string(batch_size) +
            " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (threshold_max_value < 0 || threshold_max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(threshold_max_value));
    }

    EnhancedGraphConfig config;
    config.gaussian_enabled = gaussian_enabled;
    if (gaussian_enabled) {
        if (gaussian_coeffs_1d.ndim() != 1) {
            throw std::invalid_argument("gaussian_coeffs_1d must be a 1-D float array [k]");
        }
        config.gaussian_coeffs_1d = gaussian_coeffs_1d.data();
        config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs_1d.shape(0));
        if (config.gaussian_kernel_size != 3 && config.gaussian_kernel_size != 5 &&
            config.gaussian_kernel_size != 7 && config.gaussian_kernel_size != 9) {
            throw std::invalid_argument(
                "run_experimental_enhanced_cuda_graph_gpu: Specialized Gaussian only supports kernel_size in "
                "{3,5,7,9}, got " + std::to_string(config.gaussian_kernel_size));
        }
    }

    config.median_enabled = median_enabled;

    config.sobel_enabled = sobel_enabled;
    config.sobel_mode = sobel_mode;
    if (sobel_enabled && (sobel_mode < 0 || sobel_mode > 3)) {
        throw std::invalid_argument("run_experimental_enhanced_cuda_graph_gpu: invalid sobel_mode " + std::to_string(sobel_mode));
    }

    config.laplacian_enabled = laplacian_enabled;
    if (laplacian_enabled) {
        if (laplacian_coeffs.ndim() != 2 || laplacian_coeffs.shape(0) != laplacian_coeffs.shape(1)) {
            throw std::invalid_argument("laplacian_coeffs must be a square 2-D array [k, k]");
        }
        config.laplacian_coeffs = laplacian_coeffs.data();
        config.laplacian_kernel_size = static_cast<int>(laplacian_coeffs.shape(0));
        if (config.laplacian_kernel_size != 3 && config.laplacian_kernel_size != 5) {
            throw std::invalid_argument(
                "run_experimental_enhanced_cuda_graph_gpu: Specialized Laplacian only supports kernel_size in "
                "{3,5}, got " + std::to_string(config.laplacian_kernel_size));
        }
    }
    config.laplacian_scale = laplacian_scale;
    config.laplacian_delta = laplacian_delta;

    config.threshold_enabled = threshold_enabled;
    config.threshold_value = static_cast<uint8_t>(threshold_value);
    config.threshold_max_value = static_cast<uint8_t>(threshold_max_value);

    EnhancedGraphScope scope = full_pipeline_scope ? EnhancedGraphScope::FullPipeline : EnhancedGraphScope::KernelsOnly;
    auto [output_bytes, timing] =
        pipeline.run(batch.data(), batch_size, height, width, config, use_graph, scope);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), output_bytes.data(), output_bytes.size());

    py::dict result;
    result["output"] = output_array;
    result["alloc_ms"] = timing.alloc_ms;
    result["h2d_ms"] = timing.h2d_ms;
    result["compute_ms"] = timing.compute_ms;
    result["d2h_ms"] = timing.d2h_ms;
    result["capture_ms"] = timing.capture_ms;
    result["instantiate_ms"] = timing.instantiate_ms;
    result["node_update_ms"] = timing.node_update_ms;
    result["total_ms"] = timing.total_ms();
    result["used_graph"] = timing.used_graph;
    result["graph_cache_hit"] = timing.graph_cache_hit;
    result["fallback_reason"] = timing.fallback_reason;
    return result;
}

// --------------------------------------------------------------------------
// Section 20F: EXPERIMENTAL ONLY. AsyncCudaPipeline splits a batch into
// chunks and pipelines H2D/compute/D2H across N buffer sets using 3
// explicit streams -- see pipeline_async.cuh's file header. use_enhanced
// selects the same kernels run_enhanced_cuda_pipeline()/
// run_basic_cuda_pipeline() use; gaussian_coeffs (2-D) is used for Basic,
// gaussian_coeffs_1d (1-D separable) for Enhanced -- exactly one is
// required depending on use_enhanced, matching this project's existing
// convention for that split (see run_experimental_persistent_pinned_binding).
// --------------------------------------------------------------------------
py::dict run_experimental_async_pipeline_binding(
    AsyncCudaPipeline& pipeline,
    py::array batch_obj,
    bool use_enhanced,
    bool gaussian_enabled, py::object gaussian_coeffs_obj, py::object gaussian_coeffs_1d_obj,
    bool median_enabled, int median_kernel_size,
    bool sobel_enabled, int sobel_mode,
    bool laplacian_enabled, py::array_t<float, py::array::c_style> laplacian_coeffs,
    float laplacian_scale, float laplacian_delta,
    bool threshold_enabled, int threshold_value, int threshold_max_value,
    int chunk_size, int num_buffers, bool use_pinned_staging) {
    if (!cuda_is_available()) {
        throw std::runtime_error("run_experimental_async_pipeline_gpu() requires a usable CUDA device; none was found.");
    }
    if (batch_obj.ndim() != 3) {
        throw std::invalid_argument(
            "run_experimental_async_pipeline_gpu expects a 3-D batch array [N, H, W]; got ndim=" +
            std::to_string(batch_obj.ndim()));
    }
    if (!batch_obj.dtype().is(py::dtype::of<uint8_t>())) {
        throw std::invalid_argument(
            "run_experimental_async_pipeline_gpu expects dtype uint8, got " +
            py::str(batch_obj.dtype()).cast<std::string>());
    }

    py::array_t<uint8_t, py::array::c_style> batch = py::array_t<uint8_t, py::array::c_style>::ensure(batch_obj);

    int batch_size = static_cast<int>(batch.shape(0));
    int height = static_cast<int>(batch.shape(1));
    int width = static_cast<int>(batch.shape(2));
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "run_experimental_async_pipeline_gpu received an empty batch: batch_size=" + std::to_string(batch_size) +
            " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (threshold_value < 0 || threshold_value > 255) {
        throw std::invalid_argument("threshold_value must be in [0, 255], got " + std::to_string(threshold_value));
    }
    if (threshold_max_value < 0 || threshold_max_value > 255) {
        throw std::invalid_argument("max_value must be in [0, 255], got " + std::to_string(threshold_max_value));
    }

    AsyncPipelineConfig config;
    config.use_enhanced = use_enhanced;
    config.gaussian_enabled = gaussian_enabled;
    py::array_t<float, py::array::c_style> gaussian_coeffs_arr, gaussian_coeffs_1d_arr;  // kept alive for .data()
    if (gaussian_enabled) {
        if (use_enhanced) {
            if (gaussian_coeffs_1d_obj.is_none()) {
                throw std::invalid_argument("run_experimental_async_pipeline_gpu: use_enhanced=True requires gaussian_coeffs_1d.");
            }
            gaussian_coeffs_1d_arr = py::array_t<float, py::array::c_style>::ensure(gaussian_coeffs_1d_obj);
            if (!gaussian_coeffs_1d_arr || gaussian_coeffs_1d_arr.ndim() != 1) {
                throw std::invalid_argument("gaussian_coeffs_1d must be a 1-D float array [k].");
            }
            config.gaussian_coeffs_1d = gaussian_coeffs_1d_arr.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs_1d_arr.shape(0));
            if (config.gaussian_kernel_size != 3 && config.gaussian_kernel_size != 5 &&
                config.gaussian_kernel_size != 7 && config.gaussian_kernel_size != 9) {
                throw std::invalid_argument(
                    "run_experimental_async_pipeline_gpu: Enhanced Gaussian only supports kernel_size in {3,5,7,9}, got " +
                    std::to_string(config.gaussian_kernel_size));
            }
        } else {
            if (gaussian_coeffs_obj.is_none()) {
                throw std::invalid_argument("run_experimental_async_pipeline_gpu: use_enhanced=False requires gaussian_coeffs.");
            }
            gaussian_coeffs_arr = py::array_t<float, py::array::c_style>::ensure(gaussian_coeffs_obj);
            if (!gaussian_coeffs_arr || gaussian_coeffs_arr.ndim() != 2 || gaussian_coeffs_arr.shape(0) != gaussian_coeffs_arr.shape(1)) {
                throw std::invalid_argument("gaussian_coeffs must be a square 2-D array [k, k]");
            }
            config.gaussian_coeffs = gaussian_coeffs_arr.data();
            config.gaussian_kernel_size = static_cast<int>(gaussian_coeffs_arr.shape(0));
        }
    }

    config.median_enabled = median_enabled;
    config.median_kernel_size = median_kernel_size;
    if (median_enabled && !use_enhanced &&
        (median_kernel_size <= 0 || median_kernel_size % 2 == 0 || median_kernel_size > kMedianBasicMaxKernelSize)) {
        throw std::invalid_argument("run_experimental_async_pipeline_gpu: invalid median_kernel_size " + std::to_string(median_kernel_size));
    }

    config.sobel_enabled = sobel_enabled;
    config.sobel_mode = sobel_mode;
    if (sobel_enabled && (sobel_mode < 0 || sobel_mode > 3)) {
        throw std::invalid_argument("run_experimental_async_pipeline_gpu: invalid sobel_mode " + std::to_string(sobel_mode));
    }

    config.laplacian_enabled = laplacian_enabled;
    if (laplacian_enabled) {
        if (laplacian_coeffs.ndim() != 2 || laplacian_coeffs.shape(0) != laplacian_coeffs.shape(1)) {
            throw std::invalid_argument("laplacian_coeffs must be a square 2-D array [k, k]");
        }
        config.laplacian_coeffs = laplacian_coeffs.data();
        config.laplacian_kernel_size = static_cast<int>(laplacian_coeffs.shape(0));
        if (use_enhanced && config.laplacian_kernel_size != 3 && config.laplacian_kernel_size != 5) {
            throw std::invalid_argument(
                "run_experimental_async_pipeline_gpu: Enhanced Laplacian only supports kernel_size in {3,5}, got " +
                std::to_string(config.laplacian_kernel_size));
        }
    }
    config.laplacian_scale = laplacian_scale;
    config.laplacian_delta = laplacian_delta;

    config.threshold_enabled = threshold_enabled;
    config.threshold_value = static_cast<uint8_t>(threshold_value);
    config.threshold_max_value = static_cast<uint8_t>(threshold_max_value);

    auto [output_bytes, timing] = pipeline.run(
        batch.data(), batch_size, height, width, config, chunk_size, num_buffers, use_pinned_staging);

    py::array_t<uint8_t> output_array({batch_size, height, width});
    std::memcpy(output_array.mutable_data(), output_bytes.data(), output_bytes.size());

    py::dict result;
    result["output"] = output_array;
    result["alloc_ms"] = timing.alloc_ms;
    result["host_stage_ms"] = timing.host_stage_ms;
    result["total_ms"] = timing.total_ms;
    result["h2d_stream_span_ms"] = timing.h2d_stream_span_ms;
    result["compute_stream_span_ms"] = timing.compute_stream_span_ms;
    result["d2h_stream_span_ms"] = timing.d2h_stream_span_ms;
    result["h2d_compute_overlap_ms"] = timing.h2d_compute_overlap_ms;
    result["compute_d2h_overlap_ms"] = timing.compute_d2h_overlap_ms;
    result["num_chunks"] = timing.num_chunks;
    result["chunk_size_used"] = timing.chunk_size_used;
    result["num_buffers_used"] = timing.num_buffers_used;
    result["used_pinned_staging"] = timing.used_pinned_staging;
    return result;
}

py::dict device_memory_info() {
    if (!cuda_is_available()) {
        throw std::runtime_error("device_memory_info() requires a usable CUDA device; none was found.");
    }
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    py::dict result;
    result["free_bytes"] = free_bytes;
    result["total_bytes"] = total_bytes;
    return result;
}

}  // namespace

PYBIND11_MODULE(xray_cuda, m) {
    m.doc() = "xray_cuda: CUDA bindings for the X-ray CUDA demo project (Sections 1 and 4A)";

    // CudaMemoryError -> Python "GpuMemoryError", a subclass of the
    // built-in MemoryError, so callers can either catch it specifically
    // or catch MemoryError generically; the original detailed C++
    // message (requested bytes vs. free/total device memory) is
    // preserved as str(exc).
    py::register_exception<CudaMemoryError>(m, "GpuMemoryError", PyExc_MemoryError);

    py::class_<GpuImage, std::shared_ptr<GpuImage>>(m, "GpuImage")
        .def_property_readonly("width", &GpuImage::width)
        .def_property_readonly("height", &GpuImage::height)
        .def_property_readonly("nbytes", &GpuImage::nbytes)
        .def_property_readonly("shape", [](const GpuImage& img) { return py::make_tuple(img.height(), img.width()); })
        .def_property_readonly("dtype", [](const GpuImage&) { return std::string("uint8"); })
        .def("__repr__", [](const GpuImage& img) {
            return "<GpuImage " + std::to_string(img.height()) + "x" + std::to_string(img.width()) +
                   " uint8, " + std::to_string(img.nbytes()) + " bytes>";
        });

    m.def("add_ints", &add_ints, "Add two integers (pure C++ sanity check)",
          py::arg("a"), py::arg("b"));

    m.def("cuda_available", &cuda_available,
          "Return True if a usable NVIDIA CUDA device is present.");

    m.def("device_info", &device_info,
          "Return a dict of properties for CUDA device 0.");

    m.def("smoke_test", &smoke_test,
          "Run a trivial CUDA kernel (input[i] + 1) and return the result "
          "plus kernel execution time in milliseconds.");

    m.def("compute_launch_config", &compute_launch_config_binding,
          "Section 23: read-only diagnostic. Returns the exact grid/block launch configuration "
          "production kernels use for a [batch_size, height, width] call, computed via the same "
          "compute_launch_grid()/default_block_dim() functions every production kernel launch "
          "already calls -- never a duplicated formula. Performs no allocation or kernel launch.",
          py::arg("width"), py::arg("height"), py::arg("batch_size"),
          py::arg("block_x") = 16, py::arg("block_y") = 16);

    m.def("upload_image", &upload_image,
          "Upload a 2-D uint8 grayscale NumPy array [H, W] to a new GpuImage. "
          "Raises ValueError for wrong ndim/dtype/empty shape, GpuMemoryError "
          "(a MemoryError subclass) if device allocation fails.",
          py::arg("image"));

    m.def("download_image", &download_image,
          "Copy a GpuImage back to a new contiguous uint8 NumPy array [H, W].",
          py::arg("gpu_image"));

    m.def("image_add_one", &image_add_one_binding,
          "Run the trivial boundary-safe +1 kernel on a GpuImage (uint8 "
          "wraparound at 255->0). Returns {'output': GpuImage, 'kernel_ms': float}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"));

    m.def("device_memory_info", &device_memory_info,
          "Return {'free_bytes': int, 'total_bytes': int} for CUDA device 0.");

    m.def("gaussian_basic_gpu", &gaussian_basic_binding,
          "Run the Basic CUDA Gaussian blur (naive, non-separable, no shared memory) "
          "on a GpuImage using precomputed [k,k] float32 coefficients (see "
          "cuda/gaussian.py::gaussian_kernel_2d). BORDER_REFLECT_101 boundary handling, "
          "matching cpu/filters.py's cv2.BORDER_DEFAULT. Returns "
          "{'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("coeffs"));

    m.def("median_basic_gpu", &median_basic_binding,
          "Run the Basic CUDA median filter (naive, no shared memory, no "
          "sorting network) on a GpuImage. BORDER_REPLICATE boundary handling, "
          "matching cv2.medianBlur's verified behavior (see cpu/filters.py::apply_median). "
          "Returns {'output': GpuImage, 'kernel_ms': float}; the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("kernel_size"));

    m.def("median_enhanced_gpu", &median_enhanced_binding,
          "Run an Enhanced (Section 7) CUDA median filter variant on a GpuImage. "
          "variant: 0=Shared (+shared-memory tiling, runtime kernel_size), "
          "1=Network3x3 (+branchless sorting network, kernel_size must be 3), "
          "2=Specialized (+compile-time kernel_size, unrolled; kernel_size in {3,5,7}). "
          "Same BORDER_REPLICATE as median_basic_gpu, exact equality (no tolerance). "
          "Returns {'output': GpuImage, 'kernel_ms': float}; the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("kernel_size"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("median_enhanced_batch_gpu", &median_enhanced_batch_binding,
          "Batched form of median_enhanced_gpu: runs the requested variant over an "
          "entire [N,H,W] uint8 batch via one kernel launch (grid.z=N). Returns "
          "{'output': np.ndarray[N,H,W] uint8, 'kernel_ms': float}.",
          py::arg("batch"), py::arg("kernel_size"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("sobel_basic_gpu", &sobel_basic_binding,
          "Run the Basic CUDA Sobel filter (naive, kernel_size=3 only, no shared "
          "memory) on a GpuImage. mode: 0=x, 1=y, 2=magnitude, 3=abs_sum "
          "(see cuda/sobel.py for the name mapping). BORDER_REFLECT_101 boundary "
          "handling, cv2.convertScaleAbs-equivalent output conversion, matching "
          "cpu/filters.py::apply_sobel. Returns {'output': GpuImage, 'kernel_ms': float}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("mode"));

    m.def("sobel_enhanced_gpu", &sobel_enhanced_binding,
          "Run an Enhanced (Section 8) CUDA Sobel filter variant on a GpuImage. "
          "variant: 0=Shared (+shared-memory tiling, same arithmetic as Basic), "
          "1=SharedConst (+coefficients in constant memory, generic 3x3 loop), "
          "2=Specialized (+compile-time mode, skips the unused derivative), "
          "3=Separable (two-pass horizontal/vertical decomposition, no shared memory). "
          "mode: 0=x, 1=y, 2=magnitude, 3=abs_sum. Same BORDER_REFLECT_101 as "
          "sobel_basic_gpu, exact equality (no tolerance). Returns {'output': GpuImage, "
          "'kernel_ms': float}; the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("mode"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("sobel_enhanced_batch_gpu", &sobel_enhanced_batch_binding,
          "Batched form of sobel_enhanced_gpu: runs the requested variant over an "
          "entire [N,H,W] uint8 batch via one launch (or one launch pair for "
          "Separable, grid.z=N). Returns {'output': np.ndarray[N,H,W] uint8, "
          "'kernel_ms': float}.",
          py::arg("batch"), py::arg("mode"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("laplacian_basic_gpu", &laplacian_basic_binding,
          "Run the Basic CUDA Laplacian filter (naive, non-separable, no shared memory) "
          "on a GpuImage using precomputed [k,k] float32 coefficients (see "
          "cuda/laplacian.py::laplacian_kernel_2d, recovered via impulse response). "
          "BORDER_REFLECT_101 boundary handling, matching cpu/filters.py's cv2.BORDER_DEFAULT. "
          "Returns {'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("coeffs"), py::arg("scale"), py::arg("delta"));

    m.def("laplacian_enhanced_gpu", &laplacian_enhanced_binding,
          "Run an Enhanced (Section 9) CUDA Laplacian filter variant on a GpuImage. "
          "variant: 0=Shared (+shared-memory tiling, runtime kernel_size, global-mem "
          "coeffs), 1=SharedConst (+constant memory, still runtime-sized), "
          "2=Specialized (+compile-time kernel_size, unrolled; kernel_size in {3,5}), "
          "3=Explicit (hand-written arithmetic for one of the three known, verified "
          "coefficient sets -- throws if coeffs match none of them). Same "
          "BORDER_REFLECT_101 as laplacian_basic_gpu, exact equality (no tolerance). "
          "Returns {'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("coeffs"), py::arg("scale"), py::arg("delta"),
          py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("laplacian_enhanced_batch_gpu", &laplacian_enhanced_batch_binding,
          "Batched form of laplacian_enhanced_gpu: runs the requested variant over an "
          "entire [N,H,W] uint8 batch via one kernel launch (grid.z=N). Returns "
          "{'output': np.ndarray[N,H,W] uint8, 'kernel_ms': float, 'coeff_upload_ms': float}.",
          py::arg("batch"), py::arg("coeffs"), py::arg("scale"), py::arg("delta"),
          py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("gaussian_enhanced_gpu", &gaussian_enhanced_binding,
          "Run an Enhanced (Section 6) CUDA Gaussian blur variant on a GpuImage using a "
          "1-D coefficient vector [k]. variant: 0=Naive (separable, global mem), "
          "1=Shared (+shared-memory tiling), 2=SharedConst (+constant memory), "
          "3=Specialized (compile-time kernel_size, unrolled; only k in {3,5,7,9}). "
          "block_x/block_y select the launch configuration (performance-only parameter). "
          "Returns {'output': GpuImage, 'coeff_upload_ms', 'horizontal_ms', 'vertical_ms', 'kernel_ms'}; "
          "the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("coeffs_1d"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("gaussian_enhanced_batch_gpu", &gaussian_enhanced_batch_binding,
          "Batched form of gaussian_enhanced_gpu: runs the requested variant over an "
          "entire [N,H,W] uint8 batch via one pair of kernel launches (grid.z=N), "
          "added to test whether batching amortizes the fixed per-launch overhead "
          "that dominates at small (224x224) single-image scale. Returns "
          "{'output': np.ndarray[N,H,W] uint8, 'coeff_upload_ms', 'horizontal_ms', "
          "'vertical_ms', 'kernel_ms'}.",
          py::arg("batch"), py::arg("coeffs_1d"), py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("threshold_basic_gpu", &threshold_basic_binding,
          "Run the Basic CUDA binary threshold (naive, pointwise) on a GpuImage: "
          "pixel > threshold_value -> max_value, else -> 0 (strict, matching "
          "cv2.THRESH_BINARY / cpu/filters.py::apply_threshold exactly). "
          "Returns {'output': GpuImage, 'kernel_ms': float}; the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("threshold_value"), py::arg("max_value"));

    m.def("threshold_enhanced_gpu", &threshold_enhanced_binding,
          "Run an Enhanced (Section 10) CUDA threshold variant on a GpuImage. "
          "variant: 0=Vectorized (uchar4, 4 pixels/thread; automatically falls back "
          "to an equivalent scalar kernel when width%4!=0), 1=MultiPixel (4 pixels/"
          "thread via a scalar unrolled loop, safe for any width). Same comparison "
          "semantics as threshold_basic_gpu, exact equality (no tolerance). Returns "
          "{'output': GpuImage, 'kernel_ms': float}; the input GpuImage is not modified.",
          py::arg("gpu_image"), py::arg("threshold_value"), py::arg("max_value"),
          py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("threshold_enhanced_batch_gpu", &threshold_enhanced_batch_binding,
          "Batched form of threshold_enhanced_gpu: runs the requested variant over an "
          "entire [N,H,W] uint8 batch via one kernel launch (grid.z=N). Returns "
          "{'output': np.ndarray[N,H,W] uint8, 'kernel_ms': float}.",
          py::arg("batch"), py::arg("threshold_value"), py::arg("max_value"),
          py::arg("variant"), py::arg("block_x"), py::arg("block_y"));

    m.def("laplacian_threshold_fused_batch_gpu", &laplacian_threshold_fused_batch_binding,
          "EXPERIMENTAL (Section 10): fused Specialized-Laplacian+Threshold in one "
          "kernel launch, no intermediate Laplacian output written to global memory. "
          "kernel_size (from coeffs.shape[0]) must be 3 or 5. Not wired into the "
          "production pipeline -- see README's Section 10 fusion experiment for the "
          "measured keep/reject decision. Returns {'output': np.ndarray[N,H,W] uint8, "
          "'kernel_ms': float}.",
          py::arg("batch"), py::arg("coeffs"), py::arg("scale"), py::arg("delta"),
          py::arg("threshold_value"), py::arg("max_value"), py::arg("block_x"), py::arg("block_y"));

    m.def("run_basic_cuda_pipeline_gpu", &run_basic_cuda_pipeline_binding,
          "Run all five Basic CUDA filters on a [N,H,W] uint8 batch in ONE native call "
          "(Section 5): H2D once, then Gaussian/Median/Sobel/Laplacian/Threshold chained "
          "via two reused ping-pong GpuImageBatch buffers (no per-stage allocation, no "
          "Python transition between stages), then D2H once. Disabled stages are skipped "
          "(timing field is None, not 0.0). Returns {'output': np.ndarray[N,H,W] uint8, "
          "'h2d_ms', 'gaussian_ms', 'median_ms', 'sobel_ms', 'laplacian_ms', "
          "'threshold_ms', 'd2h_ms', 'compute_ms', 'total_ms'}.",
          py::arg("batch"),
          py::arg("gaussian_enabled"), py::arg("gaussian_coeffs"),
          py::arg("median_enabled"), py::arg("median_kernel_size"),
          py::arg("sobel_enabled"), py::arg("sobel_mode"),
          py::arg("laplacian_enabled"), py::arg("laplacian_coeffs"),
          py::arg("laplacian_scale"), py::arg("laplacian_delta"),
          py::arg("threshold_enabled"), py::arg("threshold_value"), py::arg("threshold_max_value"),
          py::arg("gaussian_use_enhanced") = false, py::arg("gaussian_coeffs_1d") = py::none(),
          py::arg("gaussian_variant") = static_cast<int>(GaussianVariant::Specialized),
          py::arg("gaussian_block_x") = 16, py::arg("gaussian_block_y") = 16,
          py::arg("median_use_enhanced") = false,
          py::arg("median_variant") = static_cast<int>(MedianVariant::Network3x3),
          py::arg("median_block_x") = 16, py::arg("median_block_y") = 16,
          py::arg("sobel_use_enhanced") = false,
          py::arg("sobel_variant") = static_cast<int>(SobelVariant::Specialized),
          py::arg("sobel_block_x") = 16, py::arg("sobel_block_y") = 16,
          py::arg("laplacian_use_enhanced") = false,
          py::arg("laplacian_variant") = static_cast<int>(LaplacianVariant::Specialized),
          py::arg("laplacian_block_x") = 16, py::arg("laplacian_block_y") = 16,
          py::arg("threshold_use_enhanced") = false,
          py::arg("threshold_variant") = static_cast<int>(ThresholdVariant::Vectorized),
          py::arg("threshold_block_x") = 16, py::arg("threshold_block_y") = 16);

    // -- Section 20B: EXPERIMENTAL ONLY -- never called by production cuda/pipeline.py --
    py::class_<PersistentCudaPipeline>(m, "PersistentCudaPipeline",
        "EXPERIMENTAL (Section 20B): an isolated GPU-buffer-persistence + pinned-host-memory "
        "variant of the production pipeline, reachable only through this class -- never wired "
        "into run_basic_cuda_pipeline_gpu()/production cuda/pipeline.py. Construct ONE instance "
        "and call .run(...) repeatedly to exercise persistent-buffer reuse across calls; a fresh "
        "instance per call is equivalent to the non-persistent baseline. Call .release() (or let "
        "the instance go out of scope) to free any persistent GPU/pinned-host allocations.")
        .def(py::init<>())
        .def("run", &run_experimental_persistent_pinned_binding,
             "Runs one pipeline call with the requested memory strategy. Returns the same fields "
             "as run_basic_cuda_pipeline_gpu() plus 'alloc_ms', 'host_stage_ms', "
             "'used_persistent_buffers', 'used_pinned_memory', 'grew_gpu_buffers', "
             "'grew_pinned_buffers'.",
             py::arg("batch"),
             py::arg("gaussian_enabled"), py::arg("gaussian_coeffs"),
             py::arg("median_enabled"), py::arg("median_kernel_size"),
             py::arg("sobel_enabled"), py::arg("sobel_mode"),
             py::arg("laplacian_enabled"), py::arg("laplacian_coeffs"),
             py::arg("laplacian_scale"), py::arg("laplacian_delta"),
             py::arg("threshold_enabled"), py::arg("threshold_value"), py::arg("threshold_max_value"),
             py::arg("gaussian_use_enhanced") = false, py::arg("gaussian_coeffs_1d") = py::none(),
             py::arg("gaussian_variant") = static_cast<int>(GaussianVariant::Specialized),
             py::arg("gaussian_block_x") = 16, py::arg("gaussian_block_y") = 16,
             py::arg("median_use_enhanced") = false,
             py::arg("median_variant") = static_cast<int>(MedianVariant::Network3x3),
             py::arg("median_block_x") = 16, py::arg("median_block_y") = 16,
             py::arg("sobel_use_enhanced") = false,
             py::arg("sobel_variant") = static_cast<int>(SobelVariant::Specialized),
             py::arg("sobel_block_x") = 16, py::arg("sobel_block_y") = 16,
             py::arg("laplacian_use_enhanced") = false,
             py::arg("laplacian_variant") = static_cast<int>(LaplacianVariant::Specialized),
             py::arg("laplacian_block_x") = 16, py::arg("laplacian_block_y") = 16,
             py::arg("threshold_use_enhanced") = false,
             py::arg("threshold_variant") = static_cast<int>(ThresholdVariant::Vectorized),
             py::arg("threshold_block_x") = 16, py::arg("threshold_block_y") = 16,
             py::arg("use_persistent_buffers") = false, py::arg("use_pinned_memory") = false)
        .def("release", &PersistentCudaPipeline::release,
             "Frees any persistent GPU/pinned-host allocations. Idempotent -- safe to call multiple "
             "times. After this call, .run() raises RuntimeError.")
        .def_property_readonly("is_released", &PersistentCudaPipeline::is_released,
             "True once release() has been called; .run() will raise RuntimeError.");

    // -- Section 20D: EXPERIMENTAL ONLY -- never called by production cuda/pipeline.py --
    py::class_<CudaGraphPipeline>(m, "CudaGraphPipeline",
        "EXPERIMENTAL (Section 20D): CUDA Graph capture/replay for the five BASIC (non-Enhanced) "
        "filter kernels only -- see pipeline_cuda_graph.cuh for why Enhanced capture is not offered. "
        "Construct ONE instance and call .run(..., use_graph=True) repeatedly; graphs are captured "
        "and cached per unique (shape, enabled-stage-mask, kernel sizes, scalar parameter) "
        "configuration, keyed internally -- a new configuration recaptures once, then replays from "
        "cache. use_graph=False runs the identical kernel sequence without capture, for direct "
        "graph-vs-no-graph comparison on otherwise identical host code. Call .release() (or let the "
        "instance go out of scope) to free the stream and every cached graph's GPU buffers.")
        .def(py::init<>())
        .def("run", &run_experimental_cuda_graph_binding,
             "Runs one Basic-kernel pipeline call. Returns 'output' plus timing fields "
             "('alloc_ms', 'h2d_ms', 'compute_ms', 'd2h_ms', 'capture_ms', 'instantiate_ms', "
             "'node_update_ms', 'total_ms') and graph-diagnostic fields ('used_graph', "
             "'graph_cache_hit', 'fallback_reason' -- non-empty only when use_graph=True but capture "
             "failed for this configuration, in which case the result is still correct, computed via "
             "the non-graph fallback path).",
             py::arg("batch"),
             py::arg("gaussian_enabled"), py::arg("gaussian_coeffs"),
             py::arg("median_enabled"), py::arg("median_kernel_size"),
             py::arg("sobel_enabled"), py::arg("sobel_mode"),
             py::arg("laplacian_enabled"), py::arg("laplacian_coeffs"),
             py::arg("laplacian_scale"), py::arg("laplacian_delta"),
             py::arg("threshold_enabled"), py::arg("threshold_value"), py::arg("threshold_max_value"),
             py::arg("use_graph") = true, py::arg("full_pipeline_scope") = true)
        .def("release", &CudaGraphPipeline::release,
             "Destroys every cached graph and this pipeline's stream. Idempotent. After this call, "
             ".run() raises RuntimeError.")
        .def_property_readonly("is_released", &CudaGraphPipeline::is_released,
             "True once release() has been called; .run() will raise RuntimeError.")
        .def_property_readonly("cache_size", &CudaGraphPipeline::cache_size,
             "Number of distinct configurations currently holding a captured+instantiated graph.")
        .def("clear_cache", &CudaGraphPipeline::clear_cache,
             "Destroys all cached graphs (e.g. to measure a cold-cache capture cost again) without "
             "releasing the pipeline itself.");

    // -- Section 20E: EXPERIMENTAL ONLY -- never called by production cuda/pipeline.py --
    py::class_<CudaGraphEnhancedPipeline>(m, "CudaGraphEnhancedPipeline",
        "EXPERIMENTAL (Section 20E): CUDA Graph capture/replay for the five ENHANCED filter kernels "
        "in their production (Specialized/Network3x3/Vectorized) configuration -- see "
        "pipeline_cuda_graph_enhanced.cuh for why this uses isolated, verified-identical kernel "
        "copies rather than the production *_enhanced_dispatch() functions directly (those kernels "
        "have anonymous-namespace/internal linkage and cannot be called from another translation "
        "unit). Construct ONE instance and call .run(..., use_graph=True) repeatedly; graphs are "
        "captured and cached per unique (shape, enabled-stage-mask, kernel sizes, sobel_mode, scalar "
        "parameter) configuration. use_graph=False runs the identical kernel sequence without "
        "capture, for direct graph-vs-no-graph comparison on otherwise identical host code.")
        .def(py::init<>())
        .def("run", &run_experimental_enhanced_cuda_graph_binding,
             "Runs one Enhanced-kernel pipeline call. Returns 'output' plus timing fields and "
             "graph-diagnostic fields, same shape as CudaGraphPipeline.run()'s result.",
             py::arg("batch"),
             py::arg("gaussian_enabled"), py::arg("gaussian_coeffs_1d"),
             py::arg("median_enabled"),
             py::arg("sobel_enabled"), py::arg("sobel_mode"),
             py::arg("laplacian_enabled"), py::arg("laplacian_coeffs"),
             py::arg("laplacian_scale"), py::arg("laplacian_delta"),
             py::arg("threshold_enabled"), py::arg("threshold_value"), py::arg("threshold_max_value"),
             py::arg("use_graph") = true, py::arg("full_pipeline_scope") = true)
        .def("release", &CudaGraphEnhancedPipeline::release,
             "Destroys every cached graph and this pipeline's stream. Idempotent. After this call, "
             ".run() raises RuntimeError.")
        .def_property_readonly("is_released", &CudaGraphEnhancedPipeline::is_released,
             "True once release() has been called; .run() will raise RuntimeError.")
        .def_property_readonly("cache_size", &CudaGraphEnhancedPipeline::cache_size,
             "Number of distinct configurations currently holding a captured+instantiated graph.")
        .def("clear_cache", &CudaGraphEnhancedPipeline::clear_cache,
             "Destroys all cached graphs without releasing the pipeline itself.");

    // -- Section 20F: EXPERIMENTAL ONLY -- never called by production cuda/pipeline.py --
    py::class_<AsyncCudaPipeline>(m, "AsyncCudaPipeline",
        "EXPERIMENTAL (Section 20F): multi-stream, double/triple-buffered asynchronous pipeline. "
        "Splits one call's batch into chunks and pipelines H2D/compute/D2H across "
        "num_buffers buffer sets (2=double, 3=triple buffering) using 3 explicit non-default "
        "streams and CUDA events for cross-stream dependencies. use_enhanced selects the same "
        "kernels run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() use. "
        "use_pinned_staging=False (default) matches the real application's pageable-NumPy data "
        "path; use_pinned_staging=True stages each chunk through a pinned host buffer first "
        "(required for genuine H2D/compute overlap per research/async_pipeline_research.md), "
        "with the staging cost counted in 'host_stage_ms' and included in 'total_ms'.")
        .def(py::init<>())
        .def("run", &run_experimental_async_pipeline_binding,
             "Runs one chunked, pipelined pass over the batch. Returns 'output' plus timing and "
             "overlap-diagnostic fields (see class docstring).",
             py::arg("batch"), py::arg("use_enhanced"),
             py::arg("gaussian_enabled"), py::arg("gaussian_coeffs") = py::none(), py::arg("gaussian_coeffs_1d") = py::none(),
             py::arg("median_enabled"), py::arg("median_kernel_size") = 0,
             py::arg("sobel_enabled"), py::arg("sobel_mode"),
             py::arg("laplacian_enabled"), py::arg("laplacian_coeffs"),
             py::arg("laplacian_scale"), py::arg("laplacian_delta"),
             py::arg("threshold_enabled"), py::arg("threshold_value"), py::arg("threshold_max_value"),
             py::arg("chunk_size"), py::arg("num_buffers") = 2, py::arg("use_pinned_staging") = false)
        .def("release", &AsyncCudaPipeline::release,
             "Destroys this pipeline's 3 streams. Idempotent. After this call, .run() raises RuntimeError.")
        .def_property_readonly("is_released", &AsyncCudaPipeline::is_released,
             "True once release() has been called; .run() will raise RuntimeError.");
}
