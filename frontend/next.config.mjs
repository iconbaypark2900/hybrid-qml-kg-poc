/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ["three", "d3", "chart.js"],
  },
  async rewrites() {
    return {
      // After Next.js pages are checked, fall through to the FastAPI backend
      fallback: [
        {
          source: "/:path*",
          destination: "http://localhost:8765/:path*",
        },
      ],
    };
  },
};

export default nextConfig;
