import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
    server: {
    host: true,            // ← allows external access (0.0.0.0)
    port: 5173,            // ← optional, defaults to 5173
    strictPort: true,      // ← fail fast if port is in use
  },
})
