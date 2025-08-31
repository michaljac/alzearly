#!/bin/bash
set -Eeuo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; PURPLE='\033[0;35m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'

# Flags
FORCE_REGEN=false
SKIP_DATA_GEN=false
SKIP_PREPROCESS=true
SERVER_PORT=""         # desired; if busy we auto-pick upward
SERVER_HOST="0.0.0.0"  # inside container
PUBLISH_IP=""          # host IP to bind (-p <PUBLISH_IP>:host_port:container_port). Empty = all interfaces
TRACKER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-regen)     FORCE_REGEN=true; shift ;;
    --skip-data-gen)   SKIP_DATA_GEN=true; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
    --port)            SERVER_PORT="$2"; shift 2 ;;
    --host)            SERVER_HOST="$2"; shift 2 ;;       # container bind host (keep 0.0.0.0)
    --publish-ip)      PUBLISH_IP="$2"; shift 2 ;;        # host bind IP (e.g., 127.0.0.1 or LAN IP)
    --tracker)         TRACKER="$2"; shift 2 ;;
    --help|-h)
      cat <<EOF
${BOLD}Alzearly Training & Serving (Linux)${NC}
Usage: $0 [--force-regen] [--skip-data-gen] [--port PORT] [--host HOST] [--publish-ip IP] [--tracker TRACKER]
Examples:
  $0                       # train (interactive tracker), then serve (auto-pick port)
  $0 --tracker none        # train without tracking, then serve
  $0 --port 8010           # prefer 8010 (auto-shifts if busy)
  $0 --publish-ip 127.0.0.1         # serve only on localhost
  $0 --publish-ip 192.168.1.50      # bind to a specific NIC/IP
EOF
      exit 0 ;;
    *) echo -e "${RED}❌ Unknown option:${NC} $1"; exit 1 ;;
  esac
done

# Paths (absolute)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
WORKDIR="/workspace"

DATA_DIR_REL="../Data/alzearly"
mkdir -p "$REPO_DIR/$DATA_DIR_REL" && chmod 777 "$REPO_DIR/$DATA_DIR_REL"
DATA_DIR="$(cd "$REPO_DIR/$DATA_DIR_REL" && pwd)"

# Docker user mapping
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"; DOCKER_USER="${HOST_UID}:${HOST_GID}"

echo -e "${BOLD}${PURPLE}🧠 Alzearly Training Pipeline${NC}"
echo -e "${CYAN}========================================${NC}\n"

# Read config via Python helper (optional)
declare -A CFG
while IFS='=' read -r k v; do
  [[ -z "${k:-}" || -z "${v:-}" ]] && continue
  v="${v%\"}"; v="${v#\"}"
  CFG["$k"]="$v"
done < <(python get_config.py 2>/dev/null || true)

CONFIG_N_PATIENTS="${CFG[CONFIG_N_PATIENTS]:-3000}"
CONFIG_YEARS="${CFG[CONFIG_YEARS]:-2021,2022,2023,2024}"
CONFIG_POSITIVE_RATE="${CFG[CONFIG_POSITIVE_RATE]:-0.08}"
CONFIG_SEED="${CFG[CONFIG_SEED]:-42}"
CONFIG_OUTPUT_DIR="${CFG[CONFIG_OUTPUT_DIR]:-data/raw}"

echo -e "${BOLD}Configuration:${NC}"
echo -e "  Patients: ${CYAN}${CONFIG_N_PATIENTS}${NC}"
echo -e "  Tracker:  ${CYAN}${TRACKER:-Interactive prompt}${NC}"
echo -e "  Years:    ${CYAN}${CONFIG_YEARS}${NC}"
echo -e "  Pos rate: ${CYAN}${CONFIG_POSITIVE_RATE}${NC}"
echo -e "  Seed:     ${CYAN}${CONFIG_SEED}${NC}"
echo -e "  Output:   ${CYAN}${CONFIG_OUTPUT_DIR}${NC}\n"

# Helpers
fix_perms_dir () {
  local d="$1"
  mkdir -p "$d" && chmod 777 "$d" || true
  docker run --rm -v "$d:/target" --user 0:0 alzearly-serve:latest \
    sh -c "chown -R $HOST_UID:$HOST_GID /target && chmod -R 777 /target" >/dev/null 2>&1 || true
}

is_port_free() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -iTCP:"$port" -sTCP:LISTEN -P -n >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ! ss -ltn | awk 'NR>1{print $4}' | grep -qE "[:.]${port}$"
  elif command -v netstat >/dev/null 2>&1; then
    ! netstat -tuln 2>/dev/null | awk 'NR>2{print $4}' | grep -qE "[:.]${port}$"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$port" <<'PY'
import socket, sys
p=int(sys.argv[1]); s=socket.socket()
try: s.bind(("0.0.0.0", p))
except OSError: sys.exit(1)
else: s.close(); sys.exit(0)
PY
  else
    return 0
  fi
}

find_free_port() {
  local start="${1:-8001}"
  local tries="${2:-50}"
  local p="$start"
  for ((i=0; i<=tries; i++)); do
    if is_port_free "$p"; then echo "$p"; return 0; fi
    p=$((start + i + 1))
  done
  return 1
}

verify_artifacts () {
  local ad="$1"
  echo -e "${BLUE}📦 Step 2: Verifying artifacts${NC}"
  echo -e "${CYAN}--------------------------------${NC}\n"
  local required=("model.pkl" "feature_names.json" "threshold.json" "metrics.json")
  local missing=()
  for f in "${required[@]}"; do
    if [[ -f "$ad/$f" ]]; then
      echo -e "${GREEN}✅ $f${NC}"
    else
      echo -e "${RED}❌ $f - MISSING${NC}"
      missing+=("$f")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo -e "${GREEN}✅ All required artifacts present${NC}\n"
    echo -e "${GREEN}🎉 Training pipeline completed successfully!${NC}"
    echo -e "${BLUE}📤 Ready for model serving with:${NC} ${GREEN}python run_serve.py${NC}\n"
    return 0
  else
    echo
    echo -e "${YELLOW}⚠️  Warning: ${#missing[@]} required artifact(s) missing${NC}"
    for f in "${missing[@]}"; do echo "   - $f"; done
    echo -e "${YELLOW}💡 Check the training logs before serving${NC}\n"
    return 1
  fi
}

# Step 1: Data generation (if needed / not skipped)
echo -e "${BLUE}📁${NC} Checking ${CYAN}$DATA_DIR/featurized${NC}..."
DATA_FOUND=false
if compgen -G "$DATA_DIR/featurized/*.parquet" >/dev/null || compgen -G "$DATA_DIR/featurized/*.csv" >/dev/null; then
  DATA_FOUND=true
fi

if [[ "$DATA_FOUND" == false ]]; then
  echo -e "${RED}❌${NC} No featurized data found"
  if [[ "$SKIP_DATA_GEN" == false ]]; then
    echo -e "${YELLOW}🔄${NC} Generating data...\n"
    fix_perms_dir "$DATA_DIR/featurized"
    extra_flags=(--output-dir /Data/raw --num-patients "$CONFIG_N_PATIENTS" --seed "$CONFIG_SEED")
    [[ "$FORCE_REGEN" == true ]] && extra_flags+=(--force-regen)
    docker run --rm \
      --user "$DOCKER_USER" \
      -v "$REPO_DIR:$WORKDIR" \
      -v "$DATA_DIR:/Data" \
      -w "$WORKDIR" \
      -e PYTHONUNBUFFERED=1 \
      -e TQDM_DISABLE=1 \
      alzearly-datagen:latest \
      bash -lc "umask 0000 && python run_datagen.py ${extra_flags[*]}"
    echo -e "${GREEN}✅ Data generation completed${NC}\n"
    echo -e "${GREEN}✅ Preprocessing completed (handled by datagen)${NC}\n"
  else
    echo -e "${YELLOW}⏭️  Data generation skipped${NC}\n"
  fi
else
  if [[ "$FORCE_REGEN" == true ]]; then
    echo -e "${YELLOW}🔄 Force regenerating data...${NC}"
    fix_perms_dir "$DATA_DIR/featurized"
    docker run --rm \
      --user "$DOCKER_USER" \
      -v "$REPO_DIR:$WORKDIR" \
      -v "$DATA_DIR:/Data" \
      -w "$WORKDIR" \
      -e PYTHONUNBUFFERED=1 \
      -e TQDM_DISABLE=1 \
      alzearly-datagen:latest \
      bash -lc "umask 0000 && python run_datagen.py --output-dir /Data/raw --num-patients '$CONFIG_N_PATIENTS' --seed '$CONFIG_SEED' --force-regen"
    echo -e "${GREEN}✅ Data regeneration completed${NC}\n"
  else
    echo -e "${GREEN}✅ Found existing featurized data${NC}\n"
  fi
fi

# Step 2: Training
echo -e "${PURPLE}🤖${NC} Starting training...\n"
fix_perms_dir "$REPO_DIR/artifacts"
if [[ -n "$TRACKER" ]]; then
  docker run --rm \
    --user "$DOCKER_USER" \
    -v "$REPO_DIR:$WORKDIR" \
    -v "$DATA_DIR:/Data" \
    -w "$WORKDIR" \
    -e PYTHONUNBUFFERED=1 \
    -e TQDM_DISABLE=1 \
    alzearly-train:latest \
    bash -lc "umask 0000 && python run_training.py --tracker '$TRACKER' --config config/model.yaml"
else
  docker run -it --rm \
    --user "$DOCKER_USER" \
    -v "$REPO_DIR:$WORKDIR" \
    -v "$DATA_DIR:/Data" \
    -w "$WORKDIR" \
    -e PYTHONUNBUFFERED=1 \
    -e TQDM_DISABLE=1 \
    alzearly-train:latest \
    bash -lc "umask 0000 && python run_training.py --config config/model.yaml"
fi
echo -e "\n${GREEN}🎉${NC} Training complete!\n"

# Step 3: Verify artifacts (prefer repo artifacts/latest; fallback to data)
ART_DIR="$REPO_DIR/artifacts/latest"
[[ ! -d "$ART_DIR" && -d "$DATA_DIR/artifacts/latest" ]] && ART_DIR="$DATA_DIR/artifacts/latest"
ART_DIR_REAL="$(readlink -f "$ART_DIR" 2>/dev/null || echo "$ART_DIR")"
fix_perms_dir "$ART_DIR_REAL"

echo -e "${BLUE}📂 Using artifacts dir:${NC} $ART_DIR_REAL"
echo -e "${BLUE}📄 Listing:${NC}"
ls -la "$ART_DIR_REAL" || true
echo

if verify_artifacts "$ART_DIR_REAL"; then
  READY_TO_SERVE=true
else
  READY_TO_SERVE=false
fi

# Prepare overlay mount so /workspace/artifacts/latest always points to real artifacts
mkdir -p "$REPO_DIR/artifacts/latest" || true
SERVE_ARTIFACTS_MOUNT=""
if [[ "$ART_DIR_REAL" != "$REPO_DIR/artifacts/latest" ]]; then
  # Mount the real artifacts over the path the app expects
  SERVE_ARTIFACTS_MOUNT="-v ${ART_DIR_REAL}:/workspace/artifacts/latest:ro"
fi

# Step 4: Serve (pick a free host port; use same inside container)
if [[ "$READY_TO_SERVE" == true ]]; then
  desired_port="${SERVER_PORT:-8001}"
  chosen_port="$(find_free_port "$desired_port" 100)" || chosen_port=""
  if [[ -z "$chosen_port" ]]; then
    echo -e "${RED}❌ No free port found in range ${desired_port}..$((desired_port+100))${NC}"
    exit 2
  fi
  if [[ "$chosen_port" != "$desired_port" ]]; then
    echo -e "${YELLOW}⚠️  Port ${desired_port} is busy; using ${chosen_port}${NC}"
  fi

  # Host mapping prefix (bind to specific host IP if provided)
  MAP_PREFIX=""
  if [[ -n "$PUBLISH_IP" ]]; then
    MAP_PREFIX="${PUBLISH_IP}:"
  fi

  # Detect a display host IP for the URL (fallback to hostname -I)
  DISPLAY_IP="${PUBLISH_IP}"
  if [[ -z "$DISPLAY_IP" ]]; then
    DISPLAY_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
    [[ -z "$DISPLAY_IP" ]] && DISPLAY_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -z "$DISPLAY_IP" ]] && DISPLAY_IP="localhost"
  fi

  echo -e "${CYAN}🚀 Starting API Server${NC}"
  echo -e "${BLUE}🌐 Local URL:${NC}    http://localhost:${chosen_port}/docs"
  echo -e "${BLUE}🌐 Network URL:${NC}  http://${DISPLAY_IP}:${chosen_port}/docs"
  echo -e "${YELLOW}🛑 Press Ctrl+C to stop${NC}\n"

  docker run -it --rm \
    --user "$DOCKER_USER" \
    -v "$REPO_DIR:$WORKDIR" \
    -v "$DATA_DIR:/Data" \
    $SERVE_ARTIFACTS_MOUNT \
    -w "$WORKDIR" \
    -p "${MAP_PREFIX}${chosen_port}:${chosen_port}" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    alzearly-serve:latest \
    bash -lc "umask 0000 && python run_serve.py --host '${SERVER_HOST}' --port '${chosen_port}'"
else
  echo -e "${YELLOW}⚠️  Server NOT started: required artifacts are missing${NC}"
  echo -e "${YELLOW}   Re-run training and ensure artifacts are produced.${NC}"
  exit 2
fi

