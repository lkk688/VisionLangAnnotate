#!/bin/bash

# install_edgeaitool.sh - Script to install edgeaitool from local file, compiled binary, or download from online

# Default configuration
DEFAULT_SCRIPT_URL="https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/tools/edgeaitool.sh"
DEFAULT_BINARY_URL="https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/tools/edgeaitool"
DEFAULT_INSTALL_PATH="$HOME/.local/bin/edgeaitool"

# Parse command line arguments
LOCAL_FILE=""
LOCAL_BINARY=""
SCRIPT_URL="$DEFAULT_SCRIPT_URL"
BINARY_URL="$DEFAULT_BINARY_URL"
INSTALL_PATH="$DEFAULT_INSTALL_PATH"
USE_BINARY=false
NON_INTERACTIVE=false

show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --local <file>        Install from local script file instead of downloading"
  echo "  --local-binary <file> Install from local compiled binary instead of script"
  echo "  --url <url>           URL to download the script from (default: $DEFAULT_SCRIPT_URL)"
  echo "  --binary-url <url>    URL to download the compiled binary from (default: $DEFAULT_BINARY_URL)"
  echo "  --use-binary          Use compiled binary instead of script (download or local)"
  echo "  --install-path <path> Path where to install the binary (default: $DEFAULT_INSTALL_PATH)"
  echo "  --non-interactive     Run in non-interactive mode (no prompts)"
  echo "  --help                Show this help message"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --local)
      LOCAL_FILE="$2"
      shift 2
      ;;
    --local-binary)
      LOCAL_BINARY="$2"
      USE_BINARY=true
      shift 2
      ;;
    --url)
      SCRIPT_URL="$2"
      shift 2
      ;;
    --binary-url)
      BINARY_URL="$2"
      USE_BINARY=true
      shift 2
      ;;
    --use-binary)
      USE_BINARY=true
      shift
      ;;
    --install-path)
      INSTALL_PATH="$2"
      shift 2
      ;;
    --non-interactive)
      NON_INTERACTIVE=true
      shift
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Function to install from local file (script or binary)
install_from_local() {
  local source_file="$1"
  local target_path="$2"
  local is_binary="$3"
  
  if [ ! -f "$source_file" ]; then
    echo "❌ Error: Local file not found: $source_file"
    exit 1
  fi
  
  if [ "$is_binary" = true ]; then
    echo "📦 Installing from local binary: $source_file"
  else
    echo "📦 Installing from local script: $source_file"
  fi
  
  mkdir -p "$(dirname "$target_path")"
  [ -f "$target_path" ] && cp "$target_path" "${target_path}.bak"
  cp "$source_file" "$target_path"
  chmod +x "$target_path"
  echo "✅ Installed successfully to: $target_path"
}

# Function to download and install
install_from_url() {
  local url="$1"
  local target_path="$2"
  local is_binary="$3"
  
  if [ "$is_binary" = true ]; then
    echo "⬇️ Downloading edgeaitool binary from: $url"
  else
    echo "⬇️ Downloading edgeaitool script from: $url"
  fi
  
  TEMP_FILE="$(mktemp)"
  
  if curl -fsSL "$url" -o "$TEMP_FILE"; then
    if [ "$is_binary" = true ]; then
      echo "✅ Downloaded binary."
    else
      echo "✅ Downloaded script."
    fi
    
    mkdir -p "$(dirname "$target_path")"
    [ -f "$target_path" ] && cp "$target_path" "${target_path}.bak"
    mv "$TEMP_FILE" "$target_path"
    chmod +x "$target_path"
    echo "✅ Installed successfully to: $target_path"
  else
    if [ "$is_binary" = true ]; then
      echo "❌ Failed to download binary from:"
    else
      echo "❌ Failed to download script from:"
    fi
    echo "   $url"
    rm -f "$TEMP_FILE" 2>/dev/null
    exit 1
  fi
}

# Function to update PATH if needed
update_path() {
  local bin_dir="$(dirname "$INSTALL_PATH")"
  
  # Check if the directory is already in PATH
  if ! echo "$PATH" | grep -q "$bin_dir"; then
    if [ "$NON_INTERACTIVE" = true ]; then
      echo "🛠️  Adding $bin_dir to your PATH (non-interactive mode)..."
      ADD_TO_PATH=true
    else
      echo "🛠️  Adding $bin_dir to your PATH..."
      read -p "Would you like to add it to your PATH? (y/n) " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        ADD_TO_PATH=true
      else
        ADD_TO_PATH=false
      fi
    fi

    if [ "$ADD_TO_PATH" = true ]; then
      SHELL_RC=""
      if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
      elif [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
      else
        SHELL_RC="$HOME/.profile"
      fi

      echo "export PATH=\"$bin_dir:\$PATH\"" >> "$SHELL_RC"
      echo "✅ Added to $SHELL_RC"
      echo "🔄 Sourcing $SHELL_RC..."
      # Source the shell configuration file
      if [ -n "$BASH_VERSION" ]; then
        source "$HOME/.bashrc"
      elif [ -n "$ZSH_VERSION" ]; then
        source "$HOME/.zshrc"
      else
        source "$HOME/.profile"
      fi
      echo "✅ PATH updated in current shell"
      echo "ℹ️  Note: You may need to restart your terminal or open a new one for the changes to take effect in other sessions."
    else
      echo "⚠️ Please add $bin_dir to your PATH manually to use edgeaitool."
    fi
  fi
}

# Main installation logic
if [ -n "$LOCAL_BINARY" ]; then
  # Install from local binary file
  install_from_local "$LOCAL_BINARY" "$INSTALL_PATH" true
elif [ -n "$LOCAL_FILE" ]; then
  # Install from local script file
  install_from_local "$LOCAL_FILE" "$INSTALL_PATH" false
elif [ "$USE_BINARY" = true ]; then
  # Download and install binary
  install_from_url "$BINARY_URL" "$INSTALL_PATH" true
else
  # Download and install script
  install_from_url "$SCRIPT_URL" "$INSTALL_PATH" false
fi

# Update PATH if needed
update_path

echo "✅ Installation completed. You can now run: edgeaitool"

# Usage examples
cat << EOF

# Installation examples:

# Install script from online (default):
# ./install_edgeaitool.sh

# Install compiled binary from online:
# ./install_edgeaitool.sh --use-binary

# Install from local script file:
# ./install_edgeaitool.sh --local /path/to/edgeaitool.sh

# Install from local compiled binary:
# ./install_edgeaitool.sh --local-binary /path/to/edgeaitool

# Install script from custom URL:
# ./install_edgeaitool.sh --url https://example.com/path/to/edgeaitool.sh

# Install binary from custom URL:
# ./install_edgeaitool.sh --binary-url https://example.com/path/to/edgeaitool

# Install to custom location:
# ./install_edgeaitool.sh --install-path /usr/local/bin/edgeaitool

# Non-interactive installation:
# ./install_edgeaitool.sh --non-interactive

# One-liner installation from GitHub (script):
# curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/install_edgeaitool.sh | bash

# One-liner installation from GitHub (binary):
# curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/install_edgeaitool.sh | bash -s -- --use-binary
EOF