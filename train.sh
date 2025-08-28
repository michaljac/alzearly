#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Bold text
BOLD='\033[1m'
UNDERLINE='\033[4m'

# Parse command line arguments
START_SERVER=false
PORT=""
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --serve)
            START_SERVER=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help)
            echo -e "${BOLD}${CYAN}Alzearly Training Pipeline${NC}"
            echo -e "${UNDERLINE}Usage:${NC} $0 [OPTIONS]"
            echo ""
            echo -e "${BOLD}Options:${NC}"
            echo -e "  ${GREEN}--serve${NC}          Start API server after training"
            echo -e "  ${GREEN}--port PORT${NC}      Specify port for server (auto-find if not specified)"
            echo -e "  ${GREEN}--host HOST${NC}      Specify host for server (default: 0.0.0.0)"
            echo -e "  ${GREEN}--help${NC}           Show this help message"
            echo ""
            echo -e "${BOLD}Examples:${NC}"
            echo -e "  ${YELLOW}$0${NC}                    # Train only"
            echo -e "  ${YELLOW}$0 --serve${NC}            # Train and start server"
            echo -e "  ${YELLOW}$0 --serve --port 8001${NC} # Train and start server on port 8001"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo -e "${YELLOW}💡 Use --help for usage information${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BOLD}${PURPLE}🧠 Alzearly Training Pipeline${NC}"
echo -e "${CYAN}${UNDERLINE}========================================${NC}"
echo ""

# Get the current directory and Data directory (platform-independent)
CURRENT_DIR=$(pwd)
DATA_DIR="../Data/alzearly"

# Create Data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo -e "${BLUE}📁${NC} Checking for existing data in ${CYAN}$DATA_DIR/featurized${NC}..."

# Check if featurized data exists
if [ ! -f "$DATA_DIR/featurized"/*.parquet ] && [ ! -f "$DATA_DIR/featurized"/*.csv ]; then
    echo -e "${RED}❌${NC} No featurized data found"
    echo -e "${YELLOW}🔄${NC} Generating data using datagen container..."
    echo ""
    
    # Run data generation container
    docker run --rm \
        -v "$CURRENT_DIR:/workspace" \
        -v "$DATA_DIR:/Data" \
        alzearly-datagen:latest \
        python run_datagen.py
    
    echo -e "${GREEN}✅${NC} Data generation completed"
    echo ""
else
    echo -e "${GREEN}✅${NC} Found existing featurized data"
    echo ""
fi

echo -e "${PURPLE}🤖${NC} Starting training..."
# Run training container
docker run -it --rm \
    -v "$CURRENT_DIR:/workspace" \
    -v "$DATA_DIR:/Data" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    alzearly-train:latest \
    python run_training.py

echo -e "${GREEN}🎉${NC} Training pipeline completed!"
echo ""

# Start API server if requested
if [ "$START_SERVER" = true ]; then
    echo -e "${BOLD}${CYAN}🚀 Starting API Server in Docker${NC}"
    echo -e "${CYAN}${UNDERLINE}===================================${NC}"
    echo ""
    
    # Build server command
    SERVER_CMD="python run_serve.py --host 0.0.0.0"
    if [ ! -z "$PORT" ]; then
        SERVER_CMD="$SERVER_CMD --port $PORT"
    fi
    
    echo -e "${BLUE}🌐${NC} Server command: ${CYAN}$SERVER_CMD${NC}"
    echo -e "${BLUE}📖${NC} Interactive docs will be available at: ${CYAN}http://localhost:[PORT]/docs${NC}"
    echo -e "${YELLOW}🛑${NC} Press Ctrl+C to stop the server"
    echo ""
    
    # Run server in Docker container
    docker run -it --rm \
        -v "$CURRENT_DIR:/workspace" \
        -p "${PORT:-8000}:${PORT:-8000}" \
        -e PYTHONUNBUFFERED=1 \
        -e PYTHONDONTWRITEBYTECODE=1 \
        alzearly-serve:latest \
        $SERVER_CMD
else
    echo -e "${BOLD}${YELLOW}💡 Next Steps:${NC}"
    echo -e "${YELLOW}   To start the API server, run:${NC}"
    echo -e "${GREEN}   • python run_serve.py${NC}"
    echo -e "${GREEN}   • $0 --serve${NC}"
    echo -e "${GREEN}   • docker run -it --rm -v \$(pwd):/workspace -p 8000:8000 alzearly-serve:latest python run_serve.py${NC}"
    echo ""
fi
