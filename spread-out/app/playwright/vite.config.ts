import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'remotion': path.resolve(__dirname, './src/remotion-mock.tsx')
    }
  },
  server: {
    port: 3000,
    strictPort: true
  }
});
