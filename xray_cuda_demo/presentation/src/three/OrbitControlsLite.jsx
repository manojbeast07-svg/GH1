import { useRef } from "react";
import { extend, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

extend({ OrbitControls });

// A minimal OrbitControls wrapper -- avoids pulling in @react-three/drei
// (and its own dependency tree) just for this one helper. `presetRef`
// lets parent-level camera-preset buttons (Top/Front/Perspective/Reset,
// which live outside the R3F tree as plain HTML buttons) snap the camera
// to a new position without needing their own render loop.
export function OrbitControlsLite({ presetRef, autoRotate }) {
  const { camera, gl } = useThree();
  const controls = useRef();

  useFrame((_, delta) => {
    if (presetRef?.current) {
      camera.position.set(...presetRef.current);
      camera.lookAt(0, 0, 0);
      presetRef.current = null;
    }
    if (autoRotate && controls.current) {
      controls.current.autoRotate = true;
      controls.current.autoRotateSpeed = 0.6;
    } else if (controls.current) {
      controls.current.autoRotate = false;
    }
    controls.current?.update(delta);
  });

  return <orbitControls ref={controls} args={[camera, gl.domElement]} enableDamping dampingFactor={0.12} />;
}
