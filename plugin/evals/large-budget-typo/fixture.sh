#!/usr/bin/env bash
set -euo pipefail
pack="$(cd "$(dirname "$0")/../../../packs/hotel" && pwd)"
python3 "$pack/generate.py" --nights 365 --hotels 4 --linked --out base.lp > /dev/null
awk '/^group_discount_budget:/{t=1} t&&/^<= 438000$/{print "<= 43800"; t=0; next} {print}' base.lp > hotel_year.lp
rm base.lp
grep -q '^<= 43800$' hotel_year.lp
cp "$pack/model.md" .
mkdir -p .optlens/context && cp "$(dirname "$0")/../fixtures/context/"*.json .optlens/context/
