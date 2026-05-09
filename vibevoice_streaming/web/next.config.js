/** @type {import('next').NextConfig} */
const BACKEND = process.env.BACKEND_URL || "http://127.0.0.1:8080"

const nextConfig = {
  async rewrites() {
    return [
      // Proxy API + WebSocket to FastAPI backend
      { source: "/voices", destination: `${BACKEND}/voices` },
      { source: "/voices/:path*", destination: `${BACKEND}/voices/:path*` },
      { source: "/tts/:path*", destination: `${BACKEND}/tts/:path*` },
      { source: "/health", destination: `${BACKEND}/health` },
    ]
  },
}
module.exports = nextConfig
