/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ["three", "d3", "chart.js"],
  },
};

export default nextConfig;
