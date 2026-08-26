import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Section 24: standalone presentation app. Base is relative ('./') so a
// production build (npm run build) can be opened/served from any path,
// including a plain `npm run preview` or a static file host, without
// assuming it's deployed at the domain root.
export default defineConfig({
  plugins: [react()],
  base: "./",
  // host: true binds 0.0.0.0 (not just localhost) so the dev server / preview
  // is reachable from another machine on the network -- needed to view this
  // over a remote desktop session by IP instead of only on-box via localhost.
  server: { host: true, port: 5174 },
  preview: { host: true, port: 5174 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.js",
  },
});
