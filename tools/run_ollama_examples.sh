#!/bin/bash

# Script to set up and run Ollama examples

set -e

# Colors for output
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${GREEN}Ollama gpt-oss Examples Setup${NC}"
echo "============================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Error: Ollama is not installed.${NC}"
    echo "Please install Ollama from https://ollama.com"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    echo -e "${YELLOW}Warning: Ollama service is not running.${NC}"
    echo "Starting Ollama service..."
    ollama serve &
    # Wait for Ollama to start
    sleep 3
    
    # Check again
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        echo -e "${RED}Error: Failed to start Ollama service.${NC}"
        echo "Please start Ollama manually and try again."
        exit 1
    fi
    echo -e "${GREEN}Ollama service started successfully.${NC}"
fi

# Check if model is pulled
MODEL="gpt-oss:20b"
if ! ollama list | grep -q "$MODEL"; then
    echo -e "${YELLOW}Model $MODEL not found locally.${NC}"
    echo "Pulling $MODEL (this may take a while)..."
    ollama pull $MODEL
    echo -e "${GREEN}Model $MODEL pulled successfully.${NC}"
else
    echo -e "${GREEN}Model $MODEL is already available.${NC}"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install openai


# Make the example script executable
chmod +x gptoss_ollama_examples.py

echo -e "\n${GREEN}Setup complete!${NC}"
echo "Run the examples with: python gptoss_ollama_examples.py"

# Ask if user wants to run the examples now
echo "Do you want to run the examples now? (y/n)"
read -r run_now
if [[ $run_now == "y" || $run_now == "Y" ]]; then
    python gptoss_ollama_examples.py
fi