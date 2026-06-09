/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* and /storage/* to the FastAPI backend so the browser hits a
  // single origin in dev. In prod, do this in your CDN / ingress instead.
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/storage/:path*", destination: `${backend}/storage/:path*` },
    ];
  },
};

export default nextConfig;
