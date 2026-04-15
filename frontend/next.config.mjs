/** @type {import('next').NextConfig} */
const apiOrigin = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8780"
).replace(/\/$/, "");

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
          destination: `${apiOrigin}/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
