#!/usr/bin/env bash
# A clean Claude Code with the optlens plugin, in one folder, leaving your own Claude Code setup alone: its own venv
# (optlens from GitHub), its own Claude config (only the optlens plugin; you log in on the first start) and a project
# folder with the hotel try-it example (packs/hotel/examples/try_week) or your own files.
#
#   scripts/sandbox.sh DIR [--ref REF] [--project PATH] [--questions FILE] [--extras LIST]
#
#   --ref        the optlens git ref to install (default: main)
#   --project    a folder or file to copy into the project instead of the example
#   --questions  a Markdown file of questions for HOW-TO.md (default: the example's questions.md)
#   --extras     optlens extras (default: scip,mcp,pyomo)
#
# Then: DIR/start.sh (extra arguments go to claude, e.g. DIR/start.sh --continue). Delete DIR to remove it all.
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac
DIR=$1; shift
REF=main EXTRAS=scip,mcp,pyomo PROJECT="" QUESTIONS=""
HERE=$(cd "$(dirname "$0")/.." && pwd)
EXAMPLE="$HERE/packs/hotel/examples/try_week"
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF=$2; shift 2 ;;
    --project) PROJECT=$2; shift 2 ;;
    --questions) QUESTIONS=$2; shift 2 ;;
    --extras) EXTRAS=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

command -v claude >/dev/null || { echo "Claude Code (claude) is not on PATH: https://claude.com/claude-code" >&2; exit 1; }
PY=""
for p in python3.13 python3.12 python3.11 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then PY=$p; break; fi
done
[ -n "$PY" ] || { echo "optlens needs Python 3.11 or later" >&2; exit 1; }
if [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then echo "$DIR exists and is not empty" >&2; exit 1; fi
mkdir -p "$DIR/project"
DIR=$(cd "$DIR" && pwd)

echo "venv: optlens[$EXTRAS] from GitHub at $REF"
"$PY" -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install -q --disable-pip-version-check "optlens[$EXTRAS] @ git+https://github.com/jjd-lab/optlens@$REF"

echo "Claude config: the optlens plugin only"
export CLAUDE_CONFIG_DIR="$DIR/claude-config"
claude plugin marketplace add jjd-lab/optlens >/dev/null
claude plugin install optlens@optlens >/dev/null

if [ -n "$PROJECT" ]; then
  cp -R "$PROJECT" "$DIR/project/"
else
  cp "$EXAMPLE/hotel_week.lp.gz" "$EXAMPLE/params.json" "$HERE/packs/hotel/model.md" "$DIR/project/"
  QUESTIONS=${QUESTIONS:-"$EXAMPLE/questions.md"}
fi
git -C "$DIR/project" init -q

cat > "$DIR/start.sh" <<'EOF'
#!/usr/bin/env bash
# This sandbox's Claude Code: its own config and the optlens in ./venv. Extra arguments go to claude.
HERE=$(cd "$(dirname "$0")" && pwd)
unset OPTLENS_SOLVER OPTLENS_CALL_LIMIT
export CLAUDE_CONFIG_DIR="$HERE/claude-config" PATH="$HERE/venv/bin:$PATH"
cd "$HERE/project" && exec claude "$@"
EOF
chmod +x "$DIR/start.sh"

{
  echo "# optlens sandbox"
  echo
  echo "Start: \`$DIR/start.sh\` (log in on the first start, trust the folder, then \`/mcp\` should show optlens"
  echo "connected). Resume later with \`$DIR/start.sh --continue\`. Your session transcripts are in"
  echo "\`claude-config/projects/\`. Delete this folder to remove everything."
  echo
  if [ -n "$QUESTIONS" ]; then cat "$QUESTIONS"; fi
} > "$DIR/HOW-TO.md"

echo
echo "Ready: $DIR"
echo "  start:     $DIR/start.sh"
echo "  questions: $DIR/HOW-TO.md"
