import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies the API to the bento container so `npm run dev` and the
// nginx-served production build behave identically from the browser's point of
// view. proxy_buffering off matters for /ask/stream: with buffering on, nginx
// holds the SSE chunks and tokens arrive in one burst at the end.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
