import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {port: 5173, strictPort: true},
  cacheDir: '../runs/frontend-cache',
  build: {outDir: '../runs/frontend-dist', emptyOutDir: false},
});
