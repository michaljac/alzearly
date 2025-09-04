#!/usr/bin/env bash
set -euo pipefail

# --- Paths & requirements ---
DATA_DIR="${DATA_DIR:-../Data/alzearly/featurized}"  # where featurized data ends up
ART_DIR="${ART_DIR:-artifacts/latest}"               # where trained artifacts live
REQ_ART=(model.pkl feature_names.json run_log.json)

# --- Helpers ---
have_data() {
  [[ -d "$DATA_DIR" ]] && [[ -n "$(ls -A "$DATA_DIR" 2>/dev/null || true)" ]]
}

have_artifacts() {
  for f in "${REQ_ART[@]}"; do
    [[ -f "$ART_DIR/$f" ]] || return 1
  done
  return 0
}

# --- Flags ---
RETRAIN="${RETRAIN:-0}"  # set RETRAIN=1 to force training

echo "🔎 Checking data in: $DATA_DIR"
if ! have_data; then
  echo "📦 No featurized data found — generating..."
  docker compose run --rm datagen
else
  echo "✅ Featurized data found."
  echo "🔄 Regenerate data? (y/n)"
  read -r REGEN_DATA
  if [[ "$REGEN_DATA" == "y" || "$REGEN_DATA" == "Y" ]]; then
    echo "🔄 Regenerating data..."
    docker compose run --rm datagen
  else
    echo "✅ Using existing data."
  fi
fi

if [[ "$RETRAIN" == "1" ]]; then
  echo "♻️ RETRAIN=1 — running training now..."
  docker compose run --rm training
else
  echo "🔎 Checking artifacts in: $ART_DIR"
  if have_artifacts; then
    echo "✅ Model found."
    echo "🔄 Retrain model? (y/n)"
    read -r RETRAIN_MODEL
    if [[ "$RETRAIN_MODEL" == "y" || "$RETRAIN_MODEL" == "Y" ]]; then
      echo "🔄 Retraining model..."
      docker compose run --rm training
    else
      echo "✅ Using existing model."
    fi
  else
    echo "🏋️ No model found — training..."
    docker compose run --rm training
  fi
fi

echo "🚀 Starting API server..."

# Read configuration from serve.yaml
if [ -f "config/serve.yaml" ]; then
    APP_HOST=$(grep 'app_host:' config/serve.yaml | awk '{print $2}' | tr -d '"')
    APP_PORT=$(grep 'app_port:' config/serve.yaml | awk '{print $2}' | tr -d '"')
    
    # Set default values if not found
    APP_HOST=${APP_HOST:-0.0.0.0}
    APP_PORT=${APP_PORT:-8001}
    
    echo "🌐 Using host: $APP_HOST"
    echo "🔌 Using port: $APP_PORT"
    
    # Export for docker-compose
    export APP_HOST APP_PORT
else
    echo "⚠️  config/serve.yaml not found, using defaults"
fi

docker compose up -d serve

echo "📋 Current services:"
docker compose ps

echo "✅ Done. Open the docs at http://localhost:${APP_PORT:-8001}/docs"
