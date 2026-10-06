# syntax=docker/dockerfile:1.7
# ---------- stage 1: build the SPA ----------
FROM node:20-alpine AS build

WORKDIR /app

# Dependencies first so a source-only change does not reinstall node_modules.
COPY package.json package-lock.json* ./
RUN npm install

COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

# ---------- stage 2: serve ----------
FROM nginx:1.27-alpine AS runtime

# The official image runs 20-envsubst-on-templates.sh at startup, which expands
# ${CANARY_SPLIT_PERCENT} from the environment. The config is mounted as a
# template (not conf.d) so that substitution happens; nginx's own variables use
# $name without braces precisely so envsubst leaves them alone.
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY docker-entrypoint-ui.sh /docker-entrypoint-ui.sh
RUN chmod +x /docker-entrypoint-ui.sh

ENV CANARY_SPLIT_PERCENT=0

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1/healthz || exit 1

# Overrides the image's own entrypoint: the stock one only runs envsubst, which
# cannot handle CANARY_SPLIT_PERCENT=0 (nginx rejects a 0% split).
ENTRYPOINT ["/docker-entrypoint-ui.sh"]
