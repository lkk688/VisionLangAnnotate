#!/bin/bash

# compile_edgeaitool.sh - Script to compile edgeaitool.sh into a binary using SHC
#(py312) lkk@lkk-intel13:~/Developer/VisionLangAnnotate/tools$ ./compile_edgeaitool.sh --script edgeaitool.sh --output ./

# Parse command line arguments
SCRIPT_PATH=""
OUTPUT_PATH=""
SKIP_OBFUSCATION=false

show_help() {
  echo "Usage: $0 [options] --script <script_path> --output <output_path>"
  echo "Options:"
  echo "  --script <path>       Path to the script file to compile (required)"
  echo "  --output <path>       Path where to save the compiled binary (required)"
  echo "  --skip-obfuscation    Skip script obfuscation, use base64 encoding instead"
  echo "  --help                Show this help message"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
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

# Check required arguments
if [ -z "$SCRIPT_PATH" ]; then
  echo "❌ Error: Script path is required"
  show_help
fi

if [ -z "$OUTPUT_PATH" ]; then
  echo "❌ Error: Output path is required"
  show_help
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "❌ Error: Script file not found: $SCRIPT_PATH"
  exit 1
fi

# Create a temporary directory for compilation
TEMP_DIR="$(mktemp -d)"
TEMP_COMPILED="${TEMP_DIR}/edgeaitool.sh"
cp "$SCRIPT_PATH" "$TEMP_COMPILED"
chmod +x "$TEMP_COMPILED"

echo "🔒 Hiding script source code..."

# Check if shc is installed
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

# Skip obfuscation if requested
if [ "$SKIP_OBFUSCATION" = true ]; then
  echo "🔧 Skipping SHC compilation as requested, using base64 encoding..."
  USE_BASE64=true
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
    BASE64_ENCODED=$(base64 -w 0 "$SCRIPT_PATH")
    
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
    echo "⚠️ Neither SHC nor base64 are available. Cannot obfuscate script."
    exit 1
  fi
fi

# Copy the binary to the output path
echo "📦 Copying binary to $OUTPUT_PATH"
mkdir -p "$(dirname "$OUTPUT_PATH")"
[ -f "$OUTPUT_PATH" ] && cp "$OUTPUT_PATH" "${OUTPUT_PATH}.bak"
cp "$INSTALL_BINARY" "$OUTPUT_PATH"
chmod +x "$OUTPUT_PATH"

# Clean up temporary files
rm -rf "$TEMP_DIR" 2>/dev/null

echo "✅ Compilation completed. Binary saved to: $OUTPUT_PATH"