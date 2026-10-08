#!/usr/bin/env bash
# Copies the try-week model into the empty workspace; the model stays in packs/ (the plugin folder holds text only).
set -euo pipefail
src="$(cd "$(dirname "$0")/../../../packs/hotel" && pwd)"
cp "$src/examples/try_week/hotel_week.lp.gz" "$src/examples/try_week/params.json" "$src/model.md" .
mkdir -p .optlens/context && cp "$(dirname "$0")/../fixtures/context/"*.json .optlens/context/
