#!/bin/bash

# Resolve project root (contains kg_layer/, scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Docker Installation Script for Ubuntu 24.04 (WSL2)
# Run this script with: bash install-docker.sh

set -e

echo "=========================================="
echo "Docker Installation for Ubuntu 24.04"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run as root. The script will use sudo when needed."
   exit 1
fi

echo "Step 1: Removing old Docker versions (if any)..."
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo ""
echo "Step 2: Installing prerequisites..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo ""
echo "Step 3: Adding Docker's official GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo ""
echo "Step 4: Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo ""
echo "Step 5: Installing Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo ""
echo "Step 6: Adding your user to docker group..."
sudo usermod -aG docker $USER

echo ""
echo "Step 7: Starting Docker service..."
sudo service docker start

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to:"
echo "1. Log out and log back in (or run: newgrp docker)"
echo "2. Verify installation: docker --version"
echo "3. Test Docker: docker run hello-world"
echo ""
echo "To start Docker service on WSL startup, add this to ~/.bashrc:"
echo "  sudo service docker start > /dev/null 2>&1"
echo ""

