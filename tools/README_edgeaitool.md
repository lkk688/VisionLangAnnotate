# EdgeAI Tool Installation Scripts

This directory contains scripts for installing and compiling the `edgeaitool` utility for Jetson and NVIDIA GPU devices.

## Scripts Overview

### 1. `installedgeaitool.sh`

A wrapper script that provides backward compatibility with previous versions. It can either compile or install the `edgeaitool` script, or both.

**Usage:**

```bash
./installedgeaitool.sh [options]
```

**Options:**

- `--local <file>`: Use local script file instead of downloading
- `--compile-only`: Only compile the script without installing
- `--install-only`: Only install the script without compiling
- `--non-interactive`: Run in non-interactive mode (no prompts)
- `--skip-obfuscation`: Skip script obfuscation, install as plain text
- `--help`: Show help message

**Examples:**

```bash
# Basic installation (download, compile with SHC, and install)
./installedgeaitool.sh

# Use a local script file
./installedgeaitool.sh --local /path/to/edgeaitool.sh

# Only compile the script (don't install)
./installedgeaitool.sh --compile-only --local /path/to/edgeaitool.sh

# Only install the script (don't compile)
./installedgeaitool.sh --install-only --local /path/to/edgeaitool.sh

# Skip obfuscation
./installedgeaitool.sh --skip-obfuscation

# Non-interactive mode
./installedgeaitool.sh --non-interactive
```

### 2. `compile_edgeaitool.sh`

A script dedicated to compiling the `edgeaitool.sh` script into a binary using SHC (Shell Script Compiler) or base64 encoding as a fallback.

**Usage:**

```bash
./compile_edgeaitool.sh [options] --script <script_path> --output <output_path>
```

**Options:**

- `--script <path>`: Path to the script file to compile (required)
- `--output <path>`: Path where to save the compiled binary (required)
- `--skip-obfuscation`: Skip SHC compilation, use base64 encoding instead
- `--help`: Show help message

**Examples:**

```bash
# Compile a script with SHC
./compile_edgeaitool.sh --script /path/to/edgeaitool.sh --output /path/to/output/edgeaitool

# Use base64 encoding instead of SHC
./compile_edgeaitool.sh --script /path/to/edgeaitool.sh --output /path/to/output/edgeaitool --skip-obfuscation
```

### 3. `install_edgeaitool.sh`

A script dedicated to installing the `edgeaitool` script or binary, either from a local file or by downloading from a URL.

**Usage:**

```bash
./install_edgeaitool.sh [options]
```

**Options:**

- `--local <file>`: Install from local file instead of downloading
- `--url <url>`: URL to download the script from
- `--install-path <path>`: Path where to install the binary
- `--non-interactive`: Run in non-interactive mode (no prompts)
- `--help`: Show help message

**Examples:**

```bash
# Install from default URL
./install_edgeaitool.sh

# Install from local file
./install_edgeaitool.sh --local /path/to/edgeaitool.sh

# Install from custom URL
./install_edgeaitool.sh --url https://example.com/path/to/edgeaitool.sh

# Install to custom location
./install_edgeaitool.sh --install-path /usr/local/bin/edgeaitool

# Non-interactive installation
./install_edgeaitool.sh --non-interactive
```

## One-liner Installation

You can also install `edgeaitool` using a one-liner command:

```bash
# Basic installation
curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash

# Non-interactive installation
curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash -s -- --non-interactive

# Skip obfuscation
curl -fsSL https://raw.githubusercontent.com/lkk688/VisionLangAnnotate/VisionLangAnnotateModels/installedgeaitool.sh | bash -s -- --skip-obfuscation
```

## Notes

- The scripts will automatically add the installation directory to your PATH if needed.
- If SHC is not available, the scripts will fall back to using base64 encoding for basic obfuscation.
- The compiled binary will be installed to `$HOME/.local/bin/edgeaitool` by default.