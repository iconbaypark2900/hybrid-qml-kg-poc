FROM node:20-alpine AS base

RUN npm install -g pnpm

WORKDIR /app

COPY frontend/package.json frontend/pnpm-lock.yaml ./

RUN pnpm install --frozen-lockfile

COPY frontend/ .

RUN pnpm build

EXPOSE 7860

ENV PORT=7860
ENV HOSTNAME="0.0.0.0"

CMD ["node_modules/.bin/next", "start", "-p", "7860"]
