import { defineConfig } from 'vite'
import process from 'node:process'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

const useSsl = process.env.VITE_USE_SSL === 'true'

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(useSsl ? [basicSsl()] : [])],
})
