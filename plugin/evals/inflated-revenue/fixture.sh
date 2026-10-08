#!/usr/bin/env bash
set -euo pipefail
pack="$(cd "$(dirname "$0")/../../../packs/hotel" && pwd)"
python3 "$pack/generate.py" --nights 14 --hotels 1 --linked --out base.lp --params params.json > /dev/null
awk '/^room_capacity\(h1_deluxe_d4\):/{t=1} t&&/^<= 61$/{print "<= 610"; t=0; next} {print}' base.lp > hotel_week.lp
rm base.lp
grep -q '^<= 610$' hotel_week.lp
cp "$pack/model.md" .
mkdir -p .optlens/context && cp "$(dirname "$0")/../fixtures/context/"*.json .optlens/context/
