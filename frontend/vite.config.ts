import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// In Docker the API is reachable at http://api:8000 (service name).
// Locally it's at http://localhost:8000.
// VITE_API_INTERNAL_BASE is read at Vite server (Node) startup — it is NOT
// exposed to browser code.  VITE_API_BASE is the public base used by the
// browser-side client.ts and defaults to empty string (same-origin via proxy).
const apiInternalBase = process.env.VITE_API_INTERNAL_BASE ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: apiInternalBase,
        changeOrigin: true,
      },
      '/ws': {
        target: apiInternalBase.replace(/^http/, 'ws'),
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
