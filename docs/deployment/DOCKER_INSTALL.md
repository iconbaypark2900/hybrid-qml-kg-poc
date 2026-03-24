# Docker Installation Guide for WSL2 Ubuntu

This guide will help you install Docker Engine directly in your WSL2 Ubuntu environment.

## Prerequisites

- Ubuntu 24.04 (or compatible version)
- WSL2 enabled
- Sudo access

## Installation Methods

### Method 1: Automated Script (Recommended)

Run the installation script:

```bash
chmod +x install-docker.sh
bash install-docker.sh
```

The script will:
1. Remove old Docker versions (if any)
2. Install prerequisites
3. Add Docker's official repository
4. Install Docker Engine, CLI, and plugins
5. Add your user to the docker group
6. Start the Docker service

**After running the script:**
```bash
# Log out and back in, or run:
newgrp docker

# Verify installation
docker --version
docker ps

# Test Docker
docker run hello-world
```

### Method 2: Manual Installation

If you prefer to install manually, follow these steps:

#### Step 1: Remove old Docker versions
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

#### Step 2: Install prerequisites
```bash
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

#### Step 3: Add Docker's official GPG key
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

#### Step 4: Set up Docker repository
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

#### Step 5: Install Docker Engine
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### Step 6: Add your user to docker group
```bash
sudo usermod -aG docker $USER
```

#### Step 7: Start Docker service
```bash
sudo service docker start
```

#### Step 8: Verify installation
```bash
# Log out and back in, or run:
newgrp docker

# Check version
docker --version

# Test Docker
docker run hello-world
```

## Auto-start Docker on WSL Startup

Since WSL doesn't automatically start services, add this to your `~/.bashrc`:

```bash
# Auto-start Docker service
if ! pgrep -x "dockerd" > /dev/null; then
    sudo service docker start > /dev/null 2>&1
fi
```

Or create a more sophisticated startup script:

```bash
# Add to ~/.bashrc
if command -v docker > /dev/null; then
    if ! pgrep -x "dockerd" > /dev/null; then
        echo "Starting Docker..."
        sudo service docker start > /dev/null 2>&1
    fi
fi
```

## Troubleshooting

### Permission Denied Error
If you get "permission denied" errors:
```bash
# Add user to docker group (if not already done)
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

### Docker Service Not Running
```bash
# Check status
sudo service docker status

# Start manually
sudo service docker start

# Enable auto-start (optional)
sudo systemctl enable docker  # May not work in WSL2
```

### Cannot Connect to Docker Daemon
```bash
# Ensure Docker is running
sudo service docker start

# Check if dockerd is running
ps aux | grep dockerd

# Restart Docker
sudo service docker restart
```

### WSL2 Specific Issues

**Docker daemon stops after WSL restart:**
- Add the auto-start script to `~/.bashrc` (see above)
- Or manually start: `sudo service docker start`

**Performance issues:**
- Ensure you're using WSL2 (not WSL1): `wsl --list --verbose`
- Keep your WSL2 distro updated

## Verify Installation

After installation, verify everything works:

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# List running containers
docker ps

# Run a test container
docker run hello-world

# Build a test image
docker build -f deployment/Dockerfile.featuremap -t test-image .
```

## Next Steps

Once Docker is installed, you can:

1. **Build your container:**
   ```bash
   docker build -f deployment/Dockerfile.featuremap -t hybrid-qml-kg-featuremap .
   ```

2. **Run containers:**
   ```bash
   docker run --rm -it hybrid-qml-kg-featuremap
   ```

3. **Use Docker Compose:**
   ```bash
   docker compose -f deployment/docker-compose.yml up
   ```

## Alternative: Docker Desktop with WSL Integration

If you prefer using Docker Desktop (already installed on Windows):

1. Open Docker Desktop
2. Go to **Settings** → **Resources** → **WSL Integration**
3. Enable integration for **Ubuntu**
4. Click **Apply & Restart**

This method uses Docker Desktop's engine but allows you to use `docker` commands from WSL.

## References

- [Docker Engine Installation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker WSL2 Backend](https://docs.docker.com/desktop/wsl/)
- [WSL2 Documentation](https://docs.microsoft.com/en-us/windows/wsl/)

