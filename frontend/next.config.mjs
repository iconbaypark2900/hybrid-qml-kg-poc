import bundleAnalyzer from "@next/bundle-analyzer";

const withBundleAnalyzer = bundleAnalyzer({
  enabled: process.env.ANALYZE === "true",
});

const apiOrigin = (process.env.API_ORIGIN ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig = {
  reactStrictMode: true,
  serverExternalPackages: ["three", "d3", "chart.js"],
  async rewrites() {
    return {
      fallback: [{ source: "/:path*", destination: `${apiOrigin}/:path*` }],
    };
  },
};

export default withBundleAnalyzer(nextConfig);
