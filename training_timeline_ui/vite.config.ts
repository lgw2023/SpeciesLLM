import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backendUrl = process.env.TRAINING_TIMELINE_BACKEND_URL ?? "http://127.0.0.1:8766";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: backendUrl,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
