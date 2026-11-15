import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.SORTOPS_BACKEND ?? "http://127.0.0.1:8200";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    proxy: {
      "/api": { target: backend, ws: true },
      "/urdf": backend,
    },
  },
  build: { chunkSizeWarningLimit: 1200 },
});
