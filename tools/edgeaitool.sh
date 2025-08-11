#!/bin/bash

# === edgeaitool ===
# Custom dev CLI for Jetson and NVIDIA GPU devices
SCRIPT_VERSION="v1.0.0"

# ❗ Warn if run incorrectly via `bash edgeaitool version`
if [[ "$0" == "bash" && "$1" == "${BASH_SOURCE[0]}" ]]; then
  echo "⚠️  Please run this script directly, not via 'bash'."
  echo "✅ Correct: ./edgeaitool version"
  echo "❌ Wrong: bash edgeaitool version"
  exit 1
fi

# 📟 Detect device type and hardware model
IS_JETSON=false
DEVICE_TYPE="NVIDIA GPU"

# Check for Jetson-specific hardware model
if [[ -f "/proc/device-tree/model" ]]; then
  JETSON_MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
  if [[ -n "$JETSON_MODEL" ]]; then
    IS_JETSON=true
    DEVICE_TYPE="Jetson"
    echo "🧠 Detected Jetson Model: $JETSON_MODEL"
  fi
fi

# 🧰 Detect CUDA version (works on both Jetson and desktop/server GPUs)
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep release | sed -E 's/.*release ([0-9.]+),.*/\1/')
if [[ -n "$CUDA_VERSION" ]]; then
  echo "⚙️  CUDA Version: $CUDA_VERSION"
fi

# Jetson-specific version detection
if [[ "$IS_JETSON" == "true" ]]; then
  # Detect JetPack version
  JETPACK_VERSION=$(dpkg-query --show nvidia-jetpack 2>/dev/null | awk '{print $2}')
  if [[ -n "$JETPACK_VERSION" ]]; then
    echo "📦 JetPack Version: $JETPACK_VERSION"
  fi
  
  # 🤖 Detect TensorRT version
  TENSORRT_VERSION=$(dpkg-query --show libnvinfer8 2>/dev/null | awk '{print $2}')
  if [[ -n "$TENSORRT_VERSION" ]]; then
    echo "🧠 TensorRT Version: $TENSORRT_VERSION"
  fi
else

  if command -v nvidia-smi &> /dev/null; then
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "🖥️  Detected GPU: $GPU_MODEL"
  fi
fi

# 🧠 Detect cuDNN version (works on both platforms)
if [[ -f "/usr/include/cudnn_version.h" ]]; then
  CUDNN_VERSION=$(cat /usr/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 | awk '{print $3}' | paste -sd.)
  if [[ -n "$CUDNN_VERSION" ]]; then
    echo "🧬 cuDNN Version: $CUDNN_VERSION"
  fi
fi

# Function to show a spinner during long-running operations
show_spinner() {
  local PID=$1
  local MESSAGE="$2"
  local CHARS="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  
  while kill -0 $PID 2>/dev/null; do
    for (( i=0; i<${#CHARS}; i++ )); do
      echo -ne "\r${CHARS:$i:1} $MESSAGE"
      sleep 0.2
    done
  done
  
  # Clear the spinner line
  echo -ne "\r                                                  \r"
}

# Function to pull Docker image with progress indicator
pull_with_progress() {
  local IMAGE="$1"
  local MESSAGE="${2:-Downloading... Please wait}"
  
  # Start the pull in background
  docker pull $IMAGE &
  local PID=$!
  
  # Show spinner while pulling
  show_spinner $PID "$MESSAGE"
  
  # Check if pull was successful
  wait $PID
  if [ $? -eq 0 ]; then
    echo "✅ Image downloaded successfully."
    return 0
  else
    echo "❌ Failed to download image. Please check your network connection."
    return 1
  fi
}

# Container image configuration based on device type
DOCKERHUB_USER="cmpelkk"

# Select appropriate image based on device type
if [[ "$IS_JETSON" == "true" ]]; then
  # Jetson-specific image
  IMAGE_NAME="jetson-llm"
  IMAGE_TAG="v1"
  echo "🧩 Using Jetson-specific container image"
else
  # Desktop/server GPU image
  IMAGE_NAME="nvidia-llm"
  IMAGE_TAG="v1"
  echo "🧩 Using NVIDIA GPU container image"
fi

# Define image names with different formats for flexibility
LOCAL_IMAGE="$IMAGE_NAME:$IMAGE_TAG"
FULL_LOCAL_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$IMAGE_TAG"
DEV_LOCAL_IMAGE="$IMAGE_NAME-dev:latest"
DEFAULT_REMOTE_TAG="latest"
REMOTE_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$DEFAULT_REMOTE_TAG"

#WORKSPACE_DIR="$(pwd)/workspace"
#WORKSPACE_DIR="$(pwd)"
WORKSPACE_DIR="$(realpath .)"
DEV_DIR="/Developer"
MODELS_DIR="/Developer/models"
CONTAINER_NAME="llm-dev"
# CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
#   -v $WORKSPACE_DIR:/workspace \
#   -v $MODELS_DIR:/models \
#   -v $DEV_DIR:/Developer \
#   --name $CONTAINER_NAME $IMAGE_NAME"

# CONTAINER_CMD="docker run --rm -it --runtime=nvidia --network host \
#   --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
#   -v $WORKSPACE_DIR:/workspace \
#   -v $MODELS_DIR:/models \
#   -v $DEV_DIR:/Developer \
#   --name $CONTAINER_NAME $IMAGE_NAME"
# Handle X11 forwarding for both direct and SSH scenarios
# This implementation supports both:
# 1. Direct X11 access via xhost (when running locally)
# 2. SSH X11 forwarding via .Xauthority sharing (for remote/multi-hop SSH scenarios)
#
# For multi-hop SSH X11 forwarding to work properly:
# - SSH into the server with: ssh -X user@server
# - The server should have X11Forwarding yes in /etc/ssh/sshd_config
# - For some setups, X11UseLocalhost no in /etc/ssh/sshd_config may be needed
# - The xauth package must be installed on all machines in the chain
if [ -n "$DISPLAY" ]; then
  # Try xhost approach first (works for direct access)
  xhost +local:docker >/dev/null 2>&1
  
  # For SSH X11 forwarding (especially in multi-hop scenarios)
  # Copy current .Xauthority to a location accessible by the container
  if [ -f "$HOME/.Xauthority" ]; then
    mkdir -p /tmp/docker-x11
    cp -f "$HOME/.Xauthority" /tmp/docker-x11/.Xauthority
    chmod 644 /tmp/docker-x11/.Xauthority
    echo "X11 authentication prepared for forwarding"
  else
    echo "Warning: No .Xauthority file found. X11 forwarding may not work."
  fi
  
  # Handle SSH X11 forwarding display settings
  # When connecting via SSH with X11 forwarding, DISPLAY is set to localhost:XX.0
  # Docker containers need the display in the format :XX for X11 forwarding to work
  if [[ "$DISPLAY" == localhost* ]]; then
    # Extract the display number from localhost:XX.0 format
    DISPLAY_NUM=$(echo "$DISPLAY" | cut -d ':' -f 2 | cut -d '.' -f 1)
    # Set to unix socket format for container (:XX)
    CONTAINER_DISPLAY=":$DISPLAY_NUM"
    echo "Adjusted display from $DISPLAY to $CONTAINER_DISPLAY for container"
  else
    # Use the existing DISPLAY value (already in :0 format for local X server)
    CONTAINER_DISPLAY="$DISPLAY"
  fi
else
  echo "Warning: DISPLAY not set. X11 forwarding will not work."
  CONTAINER_DISPLAY=""
fi

# Create X11 directory if it doesn't exist to prevent mount errors
mkdir -p /tmp/docker-x11
touch /tmp/docker-x11/.Xauthority 2>/dev/null || true

# Configure device-specific bindings
if [[ "$IS_JETSON" == "true" ]]; then
  # Jetson-specific bindings including tegrastats
  EXTRA_BINDS="-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/docker-x11/.Xauthority:/root/.Xauthority:ro -v /dev:/dev"
  GPU_RUNTIME="--runtime=nvidia"
else
  # Desktop/server GPU bindings
  EXTRA_BINDS="-v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/docker-x11/.Xauthority:/root/.Xauthority:ro -v /dev:/dev"
  # Use nvidia-container-toolkit for desktop/server GPUs
  GPU_RUNTIME="--gpus all"
fi

VOLUME_FLAGS="-v $WORKSPACE_DIR:/workspace -v $MODELS_DIR:/models -v $DEV_DIR:/Developer"
CREATE_CMD="docker create -it $GPU_RUNTIME --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e DISPLAY=$CONTAINER_DISPLAY \
  --name $CONTAINER_NAME $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"

# EXEC_CMD is used after ensure_container_started() function
# Executes commands inside an already running container
EXEC_CMD="docker exec -it $CONTAINER_NAME" 

# Creates and starts a new container instance, starts fresh each time (stateless)
CONTAINER_CMD="docker run --rm -it $GPU_RUNTIME --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g \
  --cap-add=NET_ADMIN --cap-add=NET_RAW --security-opt seccomp=unconfined \
  -e DISPLAY=$CONTAINER_DISPLAY \
  $VOLUME_FLAGS $EXTRA_BINDS $LOCAL_IMAGE"



ensure_container_started() {
  if ! docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "🛠️  Creating persistent container '$CONTAINER_NAME'..."
    
    # Check if image exists locally with any of the possible tags
    if docker image inspect $LOCAL_IMAGE >/dev/null 2>&1; then
      echo "✅ Found local image: $LOCAL_IMAGE"
    elif docker image inspect $FULL_LOCAL_IMAGE >/dev/null 2>&1; then
      echo "✅ Found local image: $FULL_LOCAL_IMAGE"
      # Tag it with the expected name format
      docker tag $FULL_LOCAL_IMAGE $LOCAL_IMAGE
      echo "✅ Tagged $FULL_LOCAL_IMAGE as $LOCAL_IMAGE"
    elif docker image inspect $DEV_LOCAL_IMAGE >/dev/null 2>&1; then
      echo "✅ Found local image: $DEV_LOCAL_IMAGE"
      # Tag it with the expected name format
      docker tag $DEV_LOCAL_IMAGE $LOCAL_IMAGE
      echo "✅ Tagged $DEV_LOCAL_IMAGE as $LOCAL_IMAGE"
    else
      echo "📥 Image not found locally. Pulling $REMOTE_IMAGE..."
      
      # Pull with progress indicator
      if pull_with_progress $REMOTE_IMAGE "Downloading container image... Please wait"; then
        docker tag $REMOTE_IMAGE $LOCAL_IMAGE
      else
        exit 1
      fi
    fi
    
    echo "🔧 Creating container..."
    eval "$CREATE_CMD"
  fi
  docker start $CONTAINER_NAME >/dev/null
}

show_help() {
  echo "Usage: edgeaitool [option] [args]"
  echo "Options:"
  echo "  shell             - Open a shell inside the LLM container"
  echo "  jupyter           - Start JupyterLab inside the container"
  echo "  ollama <subcmd>    - Manage Ollama in container"
  echo "      serve                - Start Ollama REST API server"
  echo "      run <model>          - Run model interactively"
  echo "      list                 - List installed models"
  echo "      pull <model>         - Pull model from registry"
  echo "      delete <model>       - Delete model from disk"
  echo "      status               - Check if REST server is running"
  echo "      ask [--model xxx]    - Ask model with auto pull/cache"
  echo "  llama             - Start llama.cpp server (port 8000)"
  echo "  fastapi           - Start a FastAPI app on port 8001"
  echo "  rag               - Launch LangChain-based RAG server"
  echo "  convert           - Convert HF model to GGUF (custom script)"
  echo "  set-hostname <n>  - Change hostname (for cloned devices)"
  echo "  setup-ssh <ghuser>- Add GitHub user's SSH key for login"
  echo "  update            - Update both script and container image"
  echo "  update-container   - Update only the Docker container image"
  echo "  update-script      - Update only this script from GitHub"
  echo "  build             - Rebuild Docker image"
  echo "  status            - Show container and service status"
  echo "  mount-nfs [host] [remote_path] [local_path]  - Mount remote NFS share using .local name"
  echo "  list              - Show all available commands"
  echo "  juiceshop         - Run OWASP Juice Shop container"
  echo "  zaproxy           - Run OWASP ZAP Proxy container"
  echo "  stop              - Stop container"
  echo "  delete            - Delete container without saving"
  echo "  help              - Show this help message"
  echo "  version           - Show script version and image version"
  echo "  publish [--tag tag] - Push local image to Docker Hub"
  echo "  commit-and-publish  - Commit container and push as image"
}

show_list() {
  echo "📦 Available Features in edgeaitool:"
  echo
  echo "  shell        → Open a shell inside the LLM container"
  echo "     ▶ edgeaitool shell"
  echo
  echo "  jupyter      → Start JupyterLab (port 8888)"
  echo "     ▶ edgeaitool jupyter"
  echo
  echo "     ▶ edgeaitool ollama serve"
  echo "     ▶ edgeaitool ollama run mistral"
  echo "     ▶ edgeaitool ollama ask --model phi3 \"Explain LLMs\""
  echo "     ▶ edgeaitool ollama pull llama3"
  echo
  echo "  llama        → Start llama.cpp REST server (port 8000)"
  echo "     ▶ edgeaitool llama"
  echo
  echo "  fastapi      → Start a FastAPI app (port 8001)"
  echo "     ▶ edgeaitool fastapi"
  echo
  echo "  rag          → Run a LangChain-based RAG server"
  echo "     ▶ edgeaitool rag"
  echo
  echo "  convert      → Convert HF model to GGUF"
  echo "     ▶ edgeaitool convert"
  echo
  echo "  run          → Run any Python script inside container"
  echo "     ▶ edgeaitool run workspace/scripts/example.py"
  echo
  echo "  set-hostname → Change device hostname (for clones)"
  echo "     ▶ edgeaitool set-hostname device-02"
  echo
  echo "  setup-ssh    → Add GitHub SSH public key for login"
  echo "     ▶ edgeaitool setup-ssh your_github_username"
  echo
  echo "  update       → Update both script and container image"
  echo "     ▶ edgeaitool update"
  echo
  echo "  update-container → Update only the Docker container image"
  echo "     ▶ edgeaitool update-container"
  echo
  echo "  update-script → Update only this script from GitHub"
  echo "     ▶ edgeaitool update-script"
  echo
  echo "  build        → Rebuild the Docker image"
  echo
  echo "  mount-nfs    → Mount an NFS share from .local hostname"
  echo "     ▶ edgeaitool mount-nfs nfs-server.local /srv/nfs/shared /mnt/nfs/shared"
  echo
  echo "  juiceshop    → Run OWASP Juice Shop container"
  echo "     ▶ edgeaitool juiceshop"
  echo
  echo "  zaproxy      → Run OWASP ZAP Proxy container"
  echo "     ▶ edgeaitool zaproxy"
  echo
  echo "  status       → Show container and service status"
  echo "     ▶ edgeaitool status"
  echo
  echo "  stop         → Stop container"
  echo "     ▶ edgeaitool stop"
  echo
  echo "  delete       → Delete container without saving"
  echo "     ▶ edgeaitool delete"
}


meshvpn() {
  LOCAL_PATH="/Developer/"
  SCRIPT_NAME="jetsonnebula.sh"
  GITHUB_RAW="https://raw.githubusercontent.com/lkk688/edgeAI/main/edgeInfra/${SCRIPT_NAME}"

  echo "[📥] Downloading ${SCRIPT_NAME} from GitHub..."
  sudo mkdir -p "$LOCAL_PATH"
  sudo curl -fsSL "$GITHUB_RAW" -o "$LOCAL_PATH/$SCRIPT_NAME"
  sudo chmod +x "$LOCAL_PATH/$SCRIPT_NAME"

  echo "[🔑] Setting token..."
  sudo "$LOCAL_PATH/$SCRIPT_NAME" set-token sjsucyberaijetsonsuper25

  echo "[⚙️] Installing Mesh VPN..."
  sudo "$LOCAL_PATH/$SCRIPT_NAME" install

  # echo "[📡] Checking Nebula status..."
  # sudo "$LOCAL_PATH/$SCRIPT_NAME" status
}

check_service() {
  local port=$1
  local name=$2
  if ss -tuln | grep -q ":$port "; then
    echo "✅ $name is running on port $port"
  else
    echo "❌ $name is not running on port $port"
  fi
}

# Main command processing
case "$1" in
  shell)
    ensure_container_started
    echo "🐚 Opening shell in container..."
    if [ -n "$2" ]; then
      $EXEC_CMD bash -c "if [[ \"\$DISPLAY\" == localhost* ]]; then DISPLAY_NUM=\$(echo \"\$DISPLAY\" | cut -d ':' -f 2 | cut -d '.' -f 1); export DISPLAY=\":\$DISPLAY_NUM\"; echo \"Fixed DISPLAY from localhost:\$DISPLAY_NUM to :\$DISPLAY_NUM inside container\"; fi; $2"
    else
      $EXEC_CMD bash -c "if [[ \"\$DISPLAY\" == localhost* ]]; then DISPLAY_NUM=\$(echo \"\$DISPLAY\" | cut -d ':' -f 2 | cut -d '.' -f 1); export DISPLAY=\":\$DISPLAY_NUM\"; echo \"Fixed DISPLAY from localhost:\$DISPLAY_NUM to :\$DISPLAY_NUM inside container\"; fi; echo \"Inside container DISPLAY=\$DISPLAY\" && bash"
    fi
    ;;
  jupyter)
    ensure_container_started
    echo "🚀 Starting JupyterLab on port 8888..."
    $EXEC_CMD bash -c "if [[ \"\$DISPLAY\" == localhost* ]]; then DISPLAY_NUM=\$(echo \"\$DISPLAY\" | cut -d ':' -f 2 | cut -d '.' -f 1); export DISPLAY=\":\$DISPLAY_NUM\"; echo \"Fixed DISPLAY from localhost:\$DISPLAY_NUM to :\$DISPLAY_NUM inside container\"; fi; jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
    ;;
  ollama)
    ensure_container_started
    shift
    SUBCMD=${1:-"serve"}
    case "$SUBCMD" in
      serve)
        echo "🚀 Starting Ollama REST API server..."
        $EXEC_CMD ollama serve
        ;;
      run)
        MODEL=${2:-"mistral"}
        echo "🤖 Running Ollama model: $MODEL"
        $EXEC_CMD ollama run $MODEL
        ;;
      list)
        echo "📋 Listing installed Ollama models..."
        $EXEC_CMD ollama list
        ;;
      pull)
        MODEL=${2:-"mistral"}
        echo "⬇️ Pulling Ollama model: $MODEL"
        $EXEC_CMD ollama pull $MODEL
        ;;
      delete)
        MODEL=${2:-"mistral"}
        echo "🗑️ Deleting Ollama model: $MODEL"
        $EXEC_CMD ollama rm $MODEL
        ;;
      status)
        ensure_container_started
        echo "🔍 Checking Ollama server status..."
        $EXEC_CMD bash -c "curl -s http://localhost:11434/api/tags || echo 'Ollama server is not running'"
        ;;
      ask)
        shift
        MODEL="mistral"
        PROMPT=""
        
        # Parse arguments
        while [[ $# -gt 0 ]]; do
          case "$1" in
            --model)
              MODEL="$2"
              shift 2
              ;;
            *)
              if [ -z "$PROMPT" ]; then
                PROMPT="$1"
              else
                PROMPT="$PROMPT $1"
              fi
              shift
              ;;
          esac
        done
        
        if [ -z "$PROMPT" ]; then
          echo "❌ Missing prompt. Usage: edgeaitool ollama ask [--model model_name] \"Your question here\""
          exit 1
        fi
        
        echo "🤖 Asking $MODEL: $PROMPT"
        $EXEC_CMD bash -c "ollama run $MODEL \"$PROMPT\" || (echo 'Model not found, pulling...' && ollama pull $MODEL && ollama run $MODEL \"$PROMPT\")"
        ;;
      ui)
        echo "🚀 Starting Ollama with Gradio UI..."
        $EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /Developer/edgeAI/jetson/ollama_gradio_ui.py"
        ;;
      *)
        echo "❌ Unknown Ollama subcommand: $SUBCMD"
        echo "Available subcommands: serve, run, list, pull, delete, status, ask, ui"
        exit 1
        ;;
    esac
    ;;
  llama)
    ensure_container_started
    echo "🚀 Starting llama.cpp server on port 8000..."
    $EXEC_CMD bash -c "cd /Developer/llama.cpp && ./server -m /models/llama-2-7b-chat.Q4_K_M.gguf --host 0.0.0.0 --port 8000"
    ;;
  juiceshop)
    echo "🧃 Starting OWASP Juice Shop on port 3000..."
    docker run --rm -it -p 3000:3000 --name juiceshop bkimminich/juice-shop
    ;;
  zaproxy)
    echo "🔍 Starting OWASP ZAP Proxy on port 8080..."
    docker run --rm -it -u zap -p 8080:8080 -p 8090:8090 -i owasp/zap2docker-stable zap-webswing.sh
    ;;
  ollama-ui)
    ensure_container_started
    echo "🚀 Starting Ollama with Gradio UI..."
    $EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /Developer/edgeAI/jetson/ollama_gradio_ui.py"
    ;;
  fastapi)
    echo "🚀 Launching FastAPI server on port 8001..."
    eval "$CONTAINER_CMD uvicorn app.main:app --host 0.0.0.0 --port 8001"
    ;;
  rag)
    echo "📚 Starting LangChain RAG demo..."
    eval "$CONTAINER_CMD python3 rag_server.py"
    ;;
  convert)
    echo "🔁 Converting HF model to GGUF..."
    eval "$CONTAINER_CMD python3 /workspace/scripts/convert_hf_to_gguf.py"
    ;;
  run)
    shift
    if [ -z "$1" ]; then
      echo "❌ Missing script path. Usage: edgeaitool run <path/to/script.py>"
      exit 1
    fi
    ensure_container_started
    echo "🐍 Running Python script: $@"
    $EXEC_CMD bash -c "if [[ \"\$DISPLAY\" == localhost* ]]; then DISPLAY_NUM=\$(echo \"\$DISPLAY\" | cut -d ':' -f 2 | cut -d '.' -f 1); export DISPLAY=\":\$DISPLAY_NUM\"; echo \"Fixed DISPLAY from localhost:\$DISPLAY_NUM to :\$DISPLAY_NUM inside container\"; fi; python3 $@"
    ;;
  mount-nfs)
    shift
    REMOTE_HOST=${1:-nfs-server.local}
    REMOTE_PATH=${2:-/srv/nfs/shared}
    MOUNT_POINT=${3:-/mnt/nfs/shared}

    echo "📡 Mounting NFS share from $REMOTE_HOST:$REMOTE_PATH to $MOUNT_POINT"

    sudo mkdir -p "$MOUNT_POINT"

    sudo mount -t nfs "$REMOTE_HOST:$REMOTE_PATH" "$MOUNT_POINT"

    if mountpoint -q "$MOUNT_POINT"; then
      echo "✅ Mounted successfully."
    else
      echo "❌ Failed to mount. Check NFS server or network."
    fi
    ;;
  build)
    echo "🛠️ Rebuilding Docker image..."
    docker build -t $IMAGE_NAME .
    ;;
  status)
    echo "📦 Docker Containers:"
    docker ps | grep "$CONTAINER_NAME" || echo "❌ Container '$CONTAINER_NAME' is not running"
    echo
    echo "🔌 Services:"
    check_service 8000 "llama.cpp server"
    check_service 8001 "FastAPI server"
    check_service 8888 "JupyterLab"
    check_service 11434 "Ollama REST API"
    ;;
  update)
    echo "🔄 Updating both script and container image..."
    
    # First update the script
    echo "⬇️ Updating edgeaitool script..."
    SCRIPT_PATH=$(realpath "$0")
    BACKUP_PATH="${SCRIPT_PATH}.bak"
    TMP_PATH="${SCRIPT_PATH}.tmp"

    cp "$SCRIPT_PATH" "$BACKUP_PATH"
    echo "⬇️ Downloading latest script..."
    curl -f#L https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/edgeaitool.sh -o "$TMP_PATH"
    chmod +x "$TMP_PATH"

    echo "✅ Script downloaded. Replacing current script..."
    mv "$TMP_PATH" "$SCRIPT_PATH"
    
    # Then update the container image
    echo "📦 Updating container image..."
    if pull_with_progress $REMOTE_IMAGE "Downloading latest container image... Please wait"; then
      LOCAL_ID=$(docker image inspect $LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
      REMOTE_ID=$(docker image inspect $REMOTE_IMAGE --format '{{.Id}}' 2>/dev/null)
      if [ "$LOCAL_ID" != "$REMOTE_ID" ]; then
        echo "📦 New version detected. Updating local image..."
        docker tag $REMOTE_IMAGE $LOCAL_IMAGE
        echo "✅ Local container updated from Docker Hub."
      else
        echo "✅ Local container is already up-to-date."
      fi
    else
      exit 1
    fi
    ;;
    
  update-script)
    echo "⬇️ Updating edgeaitool script..."
    SCRIPT_PATH=$(realpath "$0")
    BACKUP_PATH="${SCRIPT_PATH}.bak"
    TMP_PATH="${SCRIPT_PATH}.tmp"

    cp "$SCRIPT_PATH" "$BACKUP_PATH"
    echo "⬇️ Downloading latest script..."
    curl -f#L https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/edgeaitool.sh -o "$TMP_PATH"
    chmod +x "$TMP_PATH"

    echo "✅ Script downloaded. Replacing current script..."
    mv "$TMP_PATH" "$SCRIPT_PATH"

    echo "✅ Script updated."
    exit 0
    ;;
  set-hostname)
    shift
    if [ -z "$1" ]; then
      echo "❌ Missing hostname. Usage: edgeaitool set-hostname <new-hostname> [github_user]"
    else
      NEW_NAME="$1"
      GITHUB_USER="${2:-}"

      echo "🔧 Setting hostname to: $NEW_NAME"
      echo "$NEW_NAME" | sudo tee /etc/hostname > /dev/null

      echo "📝 Updating /etc/hosts..."
      sudo sed -i "s/127.0.1.1\s.*/127.0.1.1\t$NEW_NAME/" /etc/hosts

      echo "🔄 Resetting machine-id..."
      sudo truncate -s 0 /etc/machine-id
      sudo rm -f /var/lib/dbus/machine-id


      echo "🆔 Writing device ID to /etc/device-id"
      echo "$NEW_NAME" | sudo tee /etc/device-id > /dev/null
      echo "🔁 Please reboot for changes to fully apply."
    fi
    ;;
  setup-ssh)
    shift
    if [ -z "$1" ]; then
      echo "❌ Usage: edgeaitool setup-ssh <github-username>"
      exit 1
    fi
    GH_USER="$1"
    echo "🔐 Setting up SSH access for GitHub user: $GH_USER"
    mkdir -p ~/.ssh
    curl -fsSL https://github.com/$GH_USER.keys >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "✅ SSH key added. You can now login from any of $GH_USER's devices using SSH."
    ;;
  stop)
    echo "🛑 Stopping container..."
    docker stop $CONTAINER_NAME
    ;;
  delete)
    echo "🗑️ Deleting container without saving..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "✅ Container '$CONTAINER_NAME' deleted."
    ;;
  version)
    echo "🧾 edgeaitool Script Version: $SCRIPT_VERSION"
    echo "🖥️  Device Type: $DEVICE_TYPE"
    echo "🧊 Docker Image: $LOCAL_IMAGE"
    
    # Check all possible image formats
    IMAGE_ID=$(docker image inspect $LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
    if [ -z "$IMAGE_ID" ]; then
      IMAGE_ID=$(docker image inspect $FULL_LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
      if [ -n "$IMAGE_ID" ]; then
        echo "🔍 Image ID: $IMAGE_ID (as $FULL_LOCAL_IMAGE)"
      else
        IMAGE_ID=$(docker image inspect $DEV_LOCAL_IMAGE --format '{{.Id}}' 2>/dev/null)
        if [ -n "$IMAGE_ID" ]; then
          echo "🔍 Image ID: $IMAGE_ID (as $DEV_LOCAL_IMAGE)"
        else
          echo "⚠️  Image not found locally."
        fi
      fi
    else
      echo "🔍 Image ID: $IMAGE_ID"
    fi
    ;;
  meshvpn)
    meshvpn
    ;;
  publish)
    shift
    TAG="$DEFAULT_REMOTE_TAG"
    if [[ "$1" == "--tag" ]]; then
      TAG="$1"
    fi
    REMOTE_TAGGED="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
    echo "📤 Preparing to push local image '$LOCAL_IMAGE' as '$REMOTE_TAGGED'"
    if ! docker image inspect $LOCAL_IMAGE >/dev/null 2>&1; then
      echo "❌ Local image '$LOCAL_IMAGE' not found. Build it first."
      exit 1
    fi
    docker tag $LOCAL_IMAGE $REMOTE_TAGGED
    docker login || exit 1
    docker push $REMOTE_TAGGED
    echo "✅ Pushed image to Docker Hub: $REMOTE_TAGGED"
    ;;
  commit-and-publish)
    shift
    TAG="$DEFAULT_REMOTE_TAG"
    if [[ "$1" == "--tag" ]]; then
      shift
      TAG="$1"
    fi
    REMOTE_TAGGED="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
    echo "📝 Committing running container '$CONTAINER_NAME' to image '$LOCAL_IMAGE'..."
    docker commit "$CONTAINER_NAME" "$LOCAL_IMAGE"
    echo "🔖 Tagging image as '$REMOTE_TAGGED'..."
    docker tag "$LOCAL_IMAGE" "$REMOTE_TAGGED"
    docker login || exit 1
    echo "📤 Pushing to Docker Hub..."
    docker push "$REMOTE_TAGGED"
    echo "✅ Committed and pushed image: $REMOTE_TAGGED"
    ;;
  force_git_pull)
    # Set to your local repo path
    REPO_PATH="/Developer/edgeAI"
    # Confirm directory exists
    if [ ! -d "$REPO_PATH/.git" ]; then
        echo "Error: $REPO_PATH is not a valid Git repository."
        exit 1
    fi

    echo "Navigating to $REPO_PATH"
    cd "$REPO_PATH" || exit 1

    echo "Discarding all local changes and untracked files..."
    git reset --hard HEAD
    git clean -fd

    git fetch origin

    echo "Resetting local branch to match origin/main..."
    git reset --hard origin/main

    echo "Update complete. Local repository is now synced with origin/main."
    ;;
  help)
    show_help
    ;;
  list)
    show_list
    ;;
  *)
    if [ -z "$1" ]; then
      show_list
    else
      echo "❌ Unknown command: $1"
      echo "Run 'edgeaitool help' for a list of available commands."
      exit 1
    fi
    ;;
esac