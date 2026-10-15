import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend's port (dev.sh exports it); 8010 keeps clear of cad-assist's 8000.
const api = `http://127.0.0.1:${process.env.PRINT_ASSIST_PORT ?? '8010'}`

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5183,
    strictPort: true,
    proxy: { '/api': api },
  },
})
