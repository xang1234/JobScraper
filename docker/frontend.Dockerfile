# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Install deps + run Vite build. This stage is discarded after producing dist/.

FROM node:22-alpine AS builder

WORKDIR /app

# Copy lockfile first for layer caching — source changes won't bust the
# npm ci cache unless dependencies actually changed.
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci

# Copy source and build (tsc type-check + vite bundle)
COPY src/frontend/ ./
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
# Tiny nginx image serving the static build output + reverse-proxying /api/.

FROM nginx:1.27-alpine AS runtime

# curl for healthcheck
RUN apk add --no-cache curl

# Replace default nginx config with our reverse proxy + SPA config
RUN rm /etc/nginx/conf.d/default.conf
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Copy Vite build output from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
