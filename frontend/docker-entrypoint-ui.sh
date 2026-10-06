#!/bin/sh
# Render the nginx config from the template.
#
# CANARY_SPLIT_PERCENT is substituted either way, but nginx REJECTS
# split_clients at exactly "0%" ("invalid percent value"), so the ordinary
# "canary off" value of 0 cannot be rendered at all. When the split is 0 every
# region between the `# --- canary begin/end ---` markers is deleted instead.
# That is valid, unambiguous, and avoids 502s from an unresolvable canary
# upstream — which is what a nonzero-but-unreachable split would produce.
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
  envsubst '$CANARY_SPLIT_PERCENT' < "$TEMPLATE" > "$WORK"
fi

cp "$WORK" "$OUTPUT"

# Fail loudly now rather than serving a broken config.
nginx -t

exec nginx -g 'daemon off;'
