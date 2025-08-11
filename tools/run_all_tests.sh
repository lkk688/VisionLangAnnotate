#!/bin/bash

# Run All Tests for VisionLangAnnotate
# This script runs all the test scripts in sequence

set -e  # Exit on error

# ANSI color codes
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
BOLD="\033[1m"
NC="\033[0m" # No Color

echo -e "${BOLD}${BLUE}VisionLangAnnotate Test Runner${NC}"
echo -e "Starting at: $(date)"

# Make all test scripts executable
chmod +x test_docker_container.py test_vlm_components.py test_vlm_inference.py

# Function to run a test and check its result
run_test() {
    local test_name=$1
    local test_script=$2
    local test_args=$3
    
    echo -e "\n${BOLD}${BLUE}Running $test_name...${NC}"
    
    if [ -f "$test_script" ]; then
        python "$test_script" $test_args
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}$test_name completed successfully${NC}"
        else
            echo -e "${YELLOW}$test_name completed with warnings or errors${NC}"
        fi
    else
        echo -e "${RED}Error: $test_script not found${NC}"
    fi
}

# Check if we're running inside Docker
if [ -f "/.dockerenv" ]; then
    echo -e "${GREEN}Running inside Docker container${NC}"
    IN_DOCKER=true
else
    echo -e "${YELLOW}Not running inside Docker container${NC}"
    echo -e "Some tests may not work correctly outside the container"
    IN_DOCKER=false
    
    # Ask if user wants to run Docker tests
    echo -e "${BOLD}Do you want to run the Docker container tests? (y/n)${NC}"
    read -r run_docker
    
    if [[ $run_docker =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Running Docker container tests...${NC}"
        ./run_docker_tests.sh
        exit 0
    fi
fi

# Ask which tests to run
echo -e "${BOLD}Which tests do you want to run?${NC}"
echo -e "1. All tests"
echo -e "2. Basic container tests only"
echo -e "3. VLM component tests only"
echo -e "4. VLM inference tests only"
read -r test_choice

case $test_choice in
    1)
        # Run all tests
        run_test "Basic Container Tests" "test_docker_container.py"
        run_test "VLM Component Tests" "test_vlm_components.py"
        
        # For inference tests, check if we have a sample image
        SAMPLE_IMAGE=""
        if [ -d "sampledata" ]; then
            # Find the first image in the sample data directory
            SAMPLE_IMAGE=$(find "sampledata" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | head -n 1)
        fi
        
        if [ -n "$SAMPLE_IMAGE" ]; then
            run_test "VLM Inference Tests" "test_vlm_inference.py" "--image $SAMPLE_IMAGE"
        else
            run_test "VLM Inference Tests" "test_vlm_inference.py"
        fi
        ;;
    2)
        # Run basic container tests only
        run_test "Basic Container Tests" "test_docker_container.py"
        ;;
    3)
        # Run VLM component tests only
        run_test "VLM Component Tests" "test_vlm_components.py"
        ;;
    4)
        # Run VLM inference tests only
        SAMPLE_IMAGE=""
        if [ -d "sampledata" ]; then
            # Find the first image in the sample data directory
            SAMPLE_IMAGE=$(find "sampledata" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | head -n 1)
        fi
        
        if [ -n "$SAMPLE_IMAGE" ]; then
            run_test "VLM Inference Tests" "test_vlm_inference.py" "--image $SAMPLE_IMAGE"
        else
            run_test "VLM Inference Tests" "test_vlm_inference.py"
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "\n${BOLD}${GREEN}All requested tests completed!${NC}"
echo -e "See the output above for test results and any warnings or errors."
echo -e "For more information, refer to TEST_README.md"