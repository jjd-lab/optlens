#!/usr/bin/env bash
# Builds the 4-hotel, 365-night linked model (279,366 rows) from the public hotel pack, then types night 200's group
# occupancy floor as 4,800 instead of 480. About 5 seconds; the 121 MB file is too large to keep in the repository.
set -euo pipefail
pack="$(cd "$(dirname "$0")/../../../packs/hotel" && pwd)"
python3 "$pack/generate.py" --nights 365 --hotels 4 --linked --out base.lp --params params.json > /dev/null
awk '/^group_occupancy\(d200\):/{t=1} t&&/^>= 480$/{print ">= 4800"; t=0; next} {print}' base.lp > hotel_year.lp
rm base.lp
grep -q '^>= 4800$' hotel_year.lp
cp "$pack/model.md" .
mkdir -p .optlens/context && cp "$(dirname "$0")/../fixtures/context/"*.json .optlens/context/
