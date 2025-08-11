#!/bin/bash

# installedgeaitool.sh - Wrapper script for backward compatibility
# This script now serves as a wrapper that can either compile or install edgeaitool

# Default configuration
SCRIPT_URL="https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/edgeaitool.sh"
INSTALL_PATH="$HOME/.local/bin/edgeaitool"

# Parse command line arguments
NON_INTERACTIVE=false
SKIP_OBFUSCATION=false
LOCAL_FILE=""
COMPILE_ONLY=false
INSTALL_ONLY=false

show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --local <file>        Use local script file instead of downloading"
  echo "  --compile-only        Only compile the script without installing"
  echo "  --install-only        Only install the script without compiling"
  echo "  --non-interactive     Run in non-interactive mode (no prompts)"
  echo "  --skip-obfuscation    Skip script obfuscation, install as plain text"
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
    --compile-only)
      COMPILE_ONLY=true
      shift
      ;;
    --install-only)
      INSTALL_ONLY=true
      shift
      ;;
    --non-interactive)
      NON_INTERACTIVE=true
      shift
      ;;
    --skip-obfuscation)
      SKIP_OBFUSCATION=true
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

# Check for conflicting options
if [ "$COMPILE_ONLY" = true ] && [ "$INSTALL_ONLY" = true ]; then
  echo "❌ Error: Cannot use both --compile-only and --install-only together"
  exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths to the new scripts
COMPILE_SCRIPT="$SCRIPT_DIR/compile_edgeaitool.sh"
INSTALL_SCRIPT="$SCRIPT_DIR/install_edgeaitool.sh"

# Check if the new scripts exist
if [ ! -f "$COMPILE_SCRIPT" ] || [ ! -f "$INSTALL_SCRIPT" ]; then
  echo "⚠️ Warning: New script files not found. Running in legacy mode."
  
  # Include the original script content here for backward compatibility
  echo "⬇️ Downloading edgeaitool from GitHub..."
  TEMP_SCRIPT="$(mktemp)"

  if curl -fsSL "$SCRIPT_URL" -o "$TEMP_SCRIPT"; then
    echo "✅ Downloaded script."
    
    # Check if shc is installed, if not try to install it
    SHC_AVAILABLE=false
    if command -v shc &> /dev/null; then
      echo "✅ SHC (Shell Script Compiler) is already installed."
      SHC_AVAILABLE=true
    else
      echo "🔧 Attempting to install SHC (Shell Script Compiler)..."
      # Try to install SHC without asking for password first
      if command -v apt-get &> /dev/null; then
        if apt-get update && apt-get install -y shc 2>/dev/null; then
          SHC_AVAILABLE=true
        elif sudo apt-get update && sudo apt-get install -y shc; then
          SHC_AVAILABLE=true
        fi
      elif command -v yum &> /dev/null; then
        if yum install -y shc 2>/dev/null; then
          SHC_AVAILABLE=true
        elif sudo yum install -y shc; then
          SHC_AVAILABLE=true
        fi
      elif command -v dnf &> /dev/null; then
        if dnf install -y shc 2>/dev/null; then
          SHC_AVAILABLE=true
        elif sudo dnf install -y shc; then
          SHC_AVAILABLE=true
        fi
      else
        echo "⚠️ Could not install SHC automatically."
      fi
      
      # Double-check if SHC is now available
      if command -v shc &> /dev/null; then
        SHC_AVAILABLE=true
      fi
    fi
    
    # Create a temporary directory for compilation
    TEMP_DIR="$(mktemp -d)"
    TEMP_COMPILED="${TEMP_DIR}/edgeaitool.sh"
    cp "$TEMP_SCRIPT" "$TEMP_COMPILED"
    chmod +x "$TEMP_COMPILED"
    
    echo "🔒 Hiding script source code..."
    # Skip obfuscation if requested
     if [ "$SKIP_OBFUSCATION" = true ]; then
       echo "🔧 Skipping obfuscation as requested..."
       INSTALL_BINARY="$TEMP_SCRIPT"
    elif [ "$SHC_AVAILABLE" = true ]; then
      # Compile the script with SHC
      echo "🔧 Using SHC to compile the script..."
      shc -f "$TEMP_COMPILED" -o "${TEMP_DIR}/edgeaitool" -r
      if [ -f "${TEMP_DIR}/edgeaitool" ]; then
        echo "✅ Successfully compiled script with SHC."
        INSTALL_BINARY="${TEMP_DIR}/edgeaitool"
      else
        echo "⚠️ SHC compilation failed, falling back to base64 encoding."
        USE_BASE64=true
      fi
    else
      echo "⚠️ SHC not available, falling back to base64 encoding."
      USE_BASE64=true
    fi
    
    # If SHC failed or isn't available, use base64 encoding as fallback
    if [ "${USE_BASE64:-false}" = true ]; then
      if command -v base64 &> /dev/null; then
        echo "🔧 Using base64 encoding to hide script content..."
        # Create a wrapper script that decodes and executes the original script
        BASE64_ENCODED=$(base64 -w 0 "$TEMP_SCRIPT")
        
        cat > "${TEMP_DIR}/edgeaitool" << EOF
#!/bin/bash
# This script was encoded to protect its contents
# Original script: edgeaitool.sh

# Execute the decoded script
bash <(echo "$BASE64_ENCODED" | base64 -d)
EOF
        
        chmod +x "${TEMP_DIR}/edgeaitool"
        echo "✅ Successfully encoded script with base64."
        INSTALL_BINARY="${TEMP_DIR}/edgeaitool"
        echo "⚠️  Note: Base64 encoding provides basic obfuscation but is not secure against determined users."
      else
        echo "⚠️ Neither SHC nor base64 are available. Installing unobfuscated script."
        INSTALL_BINARY="$TEMP_SCRIPT"
      fi
    fi

    echo "📦 Installing to $INSTALL_PATH"
    mkdir -p "$(dirname "$INSTALL_PATH")"
    [ -f "$INSTALL_PATH" ] && cp "$INSTALL_PATH" "${INSTALL_PATH}.bak"
    mv "$INSTALL_BINARY" "$INSTALL_PATH"
    chmod +x "$INSTALL_PATH"
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR" "$TEMP_SCRIPT" 2>/dev/null

    # Auto-add to PATH if missing
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
      if [ "$NON_INTERACTIVE" = true ]; then
        echo "🛠️  Adding ~/.local/bin to your PATH (non-interactive mode)..."
        ADD_TO_PATH=true
      else
        echo "🛠️  Adding ~/.local/bin to your PATH..."
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

        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "✅ Added to $SHELL_RC"
        echo "👉 Please run: source $SHELL_RC"
      else
        echo "⚠️ Please add $HOME/.local/bin to your PATH manually to use edgeaitool."
      fi
    fi

    echo "✅ Installed successfully. You can now run: edgeaitool"
  else
    echo "❌ Failed to download script from:"
    echo "   $SCRIPT_URL"
    exit 1
  fi
  
  exit 0
fi

# Build command arguments for the new scripts
COMPILE_ARGS=""
INSTALL_ARGS=""

# Add common arguments
if [ "$NON_INTERACTIVE" = true ]; then
  INSTALL_ARGS="$INSTALL_ARGS --non-interactive"
fi

if [ "$SKIP_OBFUSCATION" = true ]; then
  COMPILE_ARGS="$COMPILE_ARGS --skip-obfuscation"
fi

# Handle local file option
if [ -n "$LOCAL_FILE" ]; then
  if [ -f "$LOCAL_FILE" ]; then
    SCRIPT_TO_USE="$LOCAL_FILE"
    INSTALL_ARGS="$INSTALL_ARGS --local $LOCAL_FILE"
  else
    echo "❌ Error: Local file not found: $LOCAL_FILE"
    exit 1
  fi
else
  # Download the script if not using local file
  echo "⬇️ Downloading edgeaitool from GitHub..."
  TEMP_SCRIPT="$(mktemp)"
  
  if curl -fsSL "$SCRIPT_URL" -o "$TEMP_SCRIPT"; then
    echo "✅ Downloaded script."
    SCRIPT_TO_USE="$TEMP_SCRIPT"
  else
    echo "❌ Failed to download script from:"
    echo "   $SCRIPT_URL"
    exit 1
  fi
fi

# Execute the appropriate script based on options
if [ "$INSTALL_ONLY" = true ]; then
  # Install only
  if [ -n "$LOCAL_FILE" ]; then
    bash "$INSTALL_SCRIPT" --local "$LOCAL_FILE" $INSTALL_ARGS
  else
    bash "$INSTALL_SCRIPT" --url "$SCRIPT_URL" $INSTALL_ARGS
  fi
elif [ "$COMPILE_ONLY" = true ]; then
  # Compile only
  TEMP_OUTPUT="${TEMP_DIR:-/tmp}/edgeaitool.bin"
  bash "$COMPILE_SCRIPT" --script "$SCRIPT_TO_USE" --output "$TEMP_OUTPUT" $COMPILE_ARGS
  echo "✅ Compiled binary saved to: $TEMP_OUTPUT"
  echo "👉 You can install it manually or use install_edgeaitool.sh"
else
  # Default: compile and install
  TEMP_DIR="$(mktemp -d)"
  TEMP_OUTPUT="${TEMP_DIR}/edgeaitool.bin"
  
  # First compile
  echo "🔧 Compiling script..."
  bash "$COMPILE_SCRIPT" --script "$SCRIPT_TO_USE" --output "$TEMP_OUTPUT" $COMPILE_ARGS
  
  # Then install the compiled binary
  echo "📦 Installing compiled binary..."
  bash "$INSTALL_SCRIPT" --local "$TEMP_OUTPUT" $INSTALL_ARGS
  
  # Clean up
  rm -rf "$TEMP_DIR" 2>/dev/null
  [ -n "$TEMP_SCRIPT" ] && rm -f "$TEMP_SCRIPT" 2>/dev/null
fi

# One-liner Installer examples:
# Basic installation:
# curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash
#
# Non-interactive installation:
# curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash -s -- --non-interactive
#
# Skip obfuscation:
# curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash -s -- --skip-obfuscation
#
# Use local file:
# ./installedgeaitool.sh --local /path/to/edgeaitool.sh
#
# Compile only:
# ./installedgeaitool.sh --compile-only --local /path/to/edgeaitool.sh
#
# Install only:
# ./installedgeaitool.sh --install-only --local /path/to/edgeaitool.sh