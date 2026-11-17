#!/bin/sh
# Render the nginx config from the template.
#
# CANARY_SPLIT_PERCENT is substituted either way, but nginx REJECTS
# split_clients at exactly "0%" ("invalid percent value"), so when split is 0
# every `# --- canary begin/end ---` block is removed and direct keepalive
# `upstream bento_backend` is used. When canary > 0, `# --- nocanary begin/end ---`
# is removed so split_clients routes traffic across stable and canary upstreams.
set -eu

TEMPLATE=/etc/nginx/templates/default.conf.template
OUTPUT=/etc/nginx/conf.d/default.conf
WORK=/tmp/rendered.conf

: "${CANARY_SPLIT_PERCENT:=0}"

if [ "$CANARY_SPLIT_PERCENT" = "0" ]; then
  echo "frontend: canary disabled (CANARY_SPLIT_PERCENT=0), removing the canary blocks"
  awk '
    /# --- canary begin/ { skip = 1; next }
    /# --- canary end/   { skip = 0; next }
    !skip
  ' "$TEMPLATE" > "$WORK"
else
  echo "frontend: canary enabled at ${CANARY_SPLIT_PERCENT}%"
  awk '
    /# --- nocanary begin/ { skip = 1; next }
    /# --- nocanary end/   { skip = 0; next }
    !skip
  ' "$TEMPLATE" | envsubst '$CANARY_SPLIT_PERCENT' > "$WORK"
fi

cp "$WORK" "$OUTPUT"

# Fail loudly now rather than serving a broken config.
nginx -t

exec nginx -g 'daemon off;'
