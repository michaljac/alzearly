#!/bin/bash
set -Eeuo pipefail

# ───────────────────────────────────────────────────────────────
# Colors & styles
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
PURPLE='\033[0;35m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'; UNDERLINE='\033[4m'

# ───────────────────────────────────────────────────────────────
# CLI args
START_SERVER=false
PORT=""
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) START_SERVER=true; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --help|-h)
      echo -e "${BOLD}${CYAN}Alzearly Training Pipeline${NC}"
      echo -e "${UNDERLINE}Usage:${NC} $0 [--serve] [--port PORT] [--host HOST]"
      exit 0
      ;;
    *) echo -e "${RED}❌ Unknown option:${NC} $1"; exit 1 ;;
  esac
done

# ───────────────────────────────────────────────────────────────
# Paths (robust)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

DATA_DIR_REL="../Data/alzearly"
mkdir -p "$REPO_DIR/$DATA_DIR_REL"; chmod 777 "$REPO_DIR/$DATA_DIR_REL"
DATA_DIR="$(cd "$REPO_DIR/$DATA_DIR_REL" && pwd)"

WORKDIR="/workspace"

# Docker user mapping (avoid root-owned files)
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DOCKER_USER="${HOST_UID}:${HOST_GID}"

echo -e "${BOLD}${PURPLE}🧠 Alzearly Training Pipeline${NC}"
echo -e "${CYAN}${UNDERLINE}========================================${NC}"
echo ""

# ───────────────────────────────────────────────────────────────
# Step 0: Prepare artifacts root & fix ownership BEFORE training
ARTIFACTS_ROOT="$REPO_DIR/artifacts"
mkdir -p "$ARTIFACTS_ROOT"; chmod 777 "$ARTIFACTS_ROOT"
# Try to fix perms quietly via a short root container (image already on your machine)
docker run --rm \
  -v "$ARTIFACTS_ROOT:/artifacts" \
  --user 0:0 \
  alzearly-serve:latest \
  sh -c "chown -R $HOST_UID:$HOST_GID /artifacts && chmod -R 777 /artifacts" >/dev/null 2>&1 || true

# ───────────────────────────────────────────────────────────────
# Step 1: Data generation (only if featurized missing)
echo -e "${BLUE}📁${NC} Checking for existing data in ${CYAN}$DATA_DIR/featurized${NC}..."
if compgen -G "$DATA_DIR/featurized/*.parquet" >/dev/null || \
   compgen -G "$DATA_DIR/featurized/*.csv" >/dev/null; then
  echo -e "${GREEN}✅ Found existing featurized data${NC}"
else
  echo -e "${RED}❌${NC} No featurized data found"
  echo -e "${YELLOW}🔄 Generating data using datagen container...${NC}"
  mkdir -p "$DATA_DIR/featurized"; chmod 777 "$DATA_DIR/featurized"
  docker run --rm \
    --user "$DOCKER_USER" \
    -v "$REPO_DIR:$WORKDIR" \
    -v "$DATA_DIR:/Data" \
    -w "$WORKDIR" \
    -e PYTHONUNBUFFERED=1 \
    -e TQDM_DISABLE=1 \
    alzearly-datagen:latest \
    sh -c "umask 0000 && python run_datagen.py"
  echo -e "${GREEN}✅ Data generation completed${NC}"
fi
echo ""

# ───────────────────────────────────────────────────────────────
# Step 2: Training (as your user; umask to allow all write)
echo -e "${PURPLE}🤖 Starting training...${NC}"
docker run -it --rm \
  --user "$DOCKER_USER" \
  -v "$REPO_DIR:$WORKDIR" \
  -v "$DATA_DIR:/Data" \
  -w "$WORKDIR" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e TQDM_DISABLE=1 \
  alzearly-train:latest \
  sh -c "umask 0000 && python run_training.py"
echo ""

# ───────────────────────────────────────────────────────────────
# Step 3: Pick the correct artifacts/latest (and show debug)
CANDIDATES=()
[ -d "$REPO_DIR/artifacts/latest" ] && CANDIDATES+=("$REPO_DIR/artifacts/latest")
[ -d "$DATA_DIR/artifacts/latest" ] && CANDIDATES+=("$DATA_DIR/artifacts/latest")

RESOLVED=()
for p in "${CANDIDATES[@]}"; do
  r="$(readlink -f "$p" 2>/dev/null || echo "$p")"
  RESOLVED+=("$r")
done

echo -e "${BLUE}🔎 Artifact locations to check:${NC}"
for i in "${!CANDIDATES[@]}"; do
  echo " - ${CANDIDATES[$i]}  ->  ${RESOLVED[$i]}"
done

ARTIFACTS_DIR_REAL=""
for r in "${RESOLVED[@]}"; do
  if [ -f "$r/model.pkl" ]; then
    ARTIFACTS_DIR_REAL="$r"; break
  fi
done
if [ -z "$ARTIFACTS_DIR_REAL" ]; then
  if [ ${#RESOLVED[@]} -gt 0 ]; then
    ARTIFACTS_DIR_REAL="${RESOLVED[0]}"
  else
    ARTIFACTS_DIR_REAL="$REPO_DIR/artifacts/latest"
  fi
fi

mkdir -p "$ARTIFACTS_DIR_REAL"; chmod 777 "$ARTIFACTS_DIR_REAL"

# Repair ownership/permissions of whatever training just created
docker run --rm \
  -v "$ARTIFACTS_DIR_REAL:/artifacts" \
  --user 0:0 \
  alzearly-serve:latest \
  sh -c "chown -R $HOST_UID:$HOST_GID /artifacts && chmod -R 777 /artifacts" >/dev/null 2>&1 || true

echo -e "${BLUE}📂 Using artifacts dir:${NC} $ARTIFACTS_DIR_REAL"
echo -e "${BLUE}📄 Listing:${NC}"
ls -la "$ARTIFACTS_DIR_REAL" || true
echo ""

echo -e "${BLUE}📦 Step 2: Verifying artifacts${NC}"
echo -e "${CYAN}--------------------------------${NC}"
echo ""

REQUIRED_FILES=("model.pkl" "feature_names.json" "threshold.json" "metrics.json")
missing=()

for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "$ARTIFACTS_DIR_REAL/$f" ]; then
    echo -e "${GREEN}✅ $f${NC}"
  else
    echo -e "${RED}❌ $f - MISSING${NC}"
    missing+=("$f")
    case "$f" in
      model.pkl)              : > "$ARTIFACTS_DIR_REAL/$f" ;;
      feature_names.json)     printf '[]\n' > "$ARTIFACTS_DIR_REAL/$f" ;;
      threshold.json)         printf '{ "threshold": 0.5 }\n' > "$ARTIFACTS_DIR_REAL/$f" ;;
      metrics.json)           printf '{ "accuracy": 0.0 }\n' > "$ARTIFACTS_DIR_REAL/$f" ;;
    esac
    chmod 777 "$ARTIFACTS_DIR_REAL/$f"
  fi
done

READY_TO_SERVE=true
if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo -e "${YELLOW}⚠️  Warning: ${#missing[@]} artifact files were missing; placeholders were created${NC}"
  for f in "${missing[@]}"; do echo "   - $f"; done
  READY_TO_SERVE=false
  echo ""
else
  echo -e "${GREEN}✅ All required artifacts present${NC}"
  echo ""
fi

# ───────────────────────────────────────────────────────────────
# Step 4: Final messages & optional serving
if [ "$READY_TO_SERVE" = true ]; then
  echo -e "${GREEN}🎉 Training pipeline completed successfully!${NC}"
  echo -e "${BLUE}📤 Ready for model serving with:${NC} ${GREEN}python run_serve.py${NC}"
else
  echo -e "${YELLOW}⚠️ Training finished, but serving is disabled due to missing artifacts${NC}"
fi
echo ""

if [ "$START_SERVER" = true ]; then
  if [ "$READY_TO_SERVE" = true ]; then
    echo -e "${BOLD}${CYAN}🚀 Starting API Server in Docker${NC}"
    SERVER_CMD="python run_serve.py --host ${HOST}"
    if [[ -n "${PORT}" ]]; then
      SERVER_CMD="$SERVER_CMD --port ${PORT}"
      PORT_MAP="-p ${PORT}:${PORT}"
      DOCS_PORT="${PORT}"
    else
      SERVER_CMD="$SERVER_CMD --port 8001"
      PORT_MAP="-p 8001:8001"
      DOCS_PORT="8001"
    fi
    docker run -it --rm \
      --user "$DOCKER_USER" \
      -v "$REPO_DIR:$WORKDIR" \
      $PORT_MAP \
      -w "$WORKDIR" \
      -e PYTHONUNBUFFERED=1 \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e TQDM_DISABLE=1 \
      alzearly-serve:latest \
      sh -c "umask 0000 && $SERVER_CMD"
  else
    echo -e "${RED}❌ Cannot start server: required artifacts not ready${NC}"
    exit 2
  fi
fi