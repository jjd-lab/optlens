#!/usr/bin/env bash
set -euo pipefail
pack="$(cd "$(dirname "$0")/../../../packs/hotel" && pwd)"
python3 "$pack/generate.py" --nights 14 --hotels 1 --linked --out base.lp > /dev/null
awk '/^room_capacity\(h1_standard_d[0-9]+\):/{t=1} t&&/^<= 116$/{print "<= 126"; t=0; next} {print}' base.lp > release_block.lp
awk '/^housekeeping\(h1_d[0-9]+\):/{t=1} t&&/^<= 110$/{print "<= 130"; t=0; next} {print}' base.lp > more_housekeeping.lp
test "$(grep -c '^<= 126$' release_block.lp)" = 14
test "$(grep -c '^<= 130$' more_housekeeping.lp)" = 13
cp "$pack/model.md" .
mkdir -p .optlens/context && cp "$(dirname "$0")/../fixtures/context/"*.json .optlens/context/
