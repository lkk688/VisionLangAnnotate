#!/bin/bash

# Run Docker Container Tests for VisionLangAnnotate
# This script builds and runs the Docker container, then executes the test script inside it

set -e  # Exit on error

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
BOLD="\033[1m"
NC="\033[0m" # No Color

echo -e "${BOLD}${BLUE}VisionLangAnnotate Docker Container Test Runner${NC}"
echo -e "Starting at: $(date)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if nvidia-docker/nvidia container toolkit is available
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Container Toolkit not detected. GPU acceleration may not be available.${NC}"
    echo -e "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check if the Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
    exit 1
fi

# Check if test script exists
if [ ! -f "test_docker_container.py" ]; then
    echo -e "${RED}Error: test_docker_container.py not found in current directory${NC}"
    exit 1
fi

# Make the test script executable
chmod +x test_docker_container.py

# Ask if user wants to build the image
echo -e "${BOLD}Do you want to build the Docker image? (y/n)${NC}"
read -r build_image

if [[ $build_image =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Building Docker image...${NC}"
    docker build -t nvidia-llm-dev .
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Docker build failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Docker image built successfully${NC}"
fi

# Create model and workspace directories if they don't exist
mkdir -p "$(pwd)/models" "$(pwd)/workspace"

# Run the container with the test script
echo -e "${BLUE}Running tests inside Docker container...${NC}"

docker run --rm -it --runtime nvidia \
  -v "$(pwd)/test_docker_container.py:/workspace/test_docker_container.py" \
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/workspace:/workspace" \
  nvidia-llm-dev python /workspace/test_docker_container.py

echo -e "${BOLD}${GREEN}Tests completed!${NC}"