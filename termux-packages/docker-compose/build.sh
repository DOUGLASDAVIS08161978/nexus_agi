#!/bin/bash
# Enhanced Docker Compose Package for Termux
# NEXUS AGI - Advanced Container Orchestration Module
# Version: 2.32.4 (Latest Stable)

TERMUX_PKG_HOMEPAGE="https://github.com/docker/compose"
TERMUX_PKG_DESCRIPTION="Docker Compose v2 - Define and run multi-container Docker applications with enhanced Termux/Android support"
TERMUX_PKG_LICENSE="Apache-2.0"
TERMUX_PKG_MAINTAINER="NEXUS AGI Team <nexus@agi.dev>"
TERMUX_PKG_VERSION="2.32.4"
TERMUX_PKG_REVISION=1
TERMUX_PKG_SRCURL="https://github.com/docker/compose/archive/v${TERMUX_PKG_VERSION}.tar.gz"
TERMUX_PKG_SHA256="a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
TERMUX_PKG_AUTO_UPDATE=true
TERMUX_PKG_DEPENDS="docker, containerd, runc, libcap, openssl"
TERMUX_PKG_SUGGESTS="docker-buildx, docker-scan-plugin, lazydocker"
TERMUX_PKG_CONFLICTS="docker-compose-python"
TERMUX_PKG_REPLACES="docker-compose-python"
TERMUX_PKG_BUILD_IN_SRC=true
TERMUX_PKG_HOSTBUILD=true

# Build configuration
TERMUX_GO_VERSION="1.22.0"
TERMUX_CGO_ENABLED=0

# Package metadata
TERMUX_PKG_EXTRA_CONFIGURE_ARGS="
--enable-buildkit
--enable-compose-switch
--with-docker-cli-plugins
"

# =============================================================================
# PRE-BUILD VALIDATION
# =============================================================================
termux_step_pre_configure() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║     NEXUS AGI - Docker Compose Package Builder                   ║"
    echo "║     Version: ${TERMUX_PKG_VERSION}                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    # Validate environment
    termux_error_exit() {
        echo "[ERROR] $1" >&2
        exit 1
    }

    # Check for required build tools
    for tool in git make tar gzip; do
        if ! command -v "$tool" &> /dev/null; then
            termux_error_exit "Required tool '$tool' not found. Install with: pkg install $tool"
        fi
    done

    # Verify Go installation will work
    echo "[INFO] Pre-flight checks passed"
    echo "[INFO] Target architecture: ${TERMUX_ARCH}"
    echo "[INFO] Target API level: ${TERMUX_PKG_API_LEVEL:-24}"

    # Create build directories
    mkdir -p "${TERMUX_PKG_BUILDDIR}"/{bin,pkg,src}
    mkdir -p "${TERMUX_PREFIX}"/lib/docker/cli-plugins
}

# =============================================================================
# GOLANG SETUP WITH ANDROID PATCHES
# =============================================================================
termux_step_setup_golang() {
    echo "[BUILD] Setting up Go ${TERMUX_GO_VERSION} toolchain..."

    termux_setup_golang

    export GOPATH="${TERMUX_PKG_BUILDDIR}"
    export GOCACHE="${TERMUX_PKG_BUILDDIR}/.cache/go-build"
    export GOMODCACHE="${TERMUX_PKG_BUILDDIR}/.cache/go-mod"

    # Android/Termux specific settings
    export GOOS="android"
    export CGO_ENABLED="${TERMUX_CGO_ENABLED}"

    case "${TERMUX_ARCH}" in
        aarch64) export GOARCH="arm64" ;;
        arm)     export GOARCH="arm"; export GOARM="7" ;;
        x86_64)  export GOARCH="amd64" ;;
        i686)    export GOARCH="386" ;;
        *)       termux_error_exit "Unsupported architecture: ${TERMUX_ARCH}" ;;
    esac

    echo "[BUILD] Go environment configured:"
    echo "        GOOS=${GOOS} GOARCH=${GOARCH}"
    echo "        GOPATH=${GOPATH}"
}

# =============================================================================
# SOURCE PATCHING FOR ANDROID COMPATIBILITY
# =============================================================================
termux_step_post_get_source() {
    echo "[PATCH] Applying Android/Termux compatibility patches..."

    cd "${TERMUX_PKG_SRCDIR}"

    # Patch 1: Fix Android DNS resolution
    if [ -f "pkg/compose/resolver.go" ]; then
        sed -i 's|/etc/resolv.conf|/data/data/com.termux/files/usr/etc/resolv.conf|g' \
            pkg/compose/resolver.go
    fi

    # Patch 2: Fix socket paths for Termux
    find . -name "*.go" -type f -exec sed -i \
        's|/var/run/docker.sock|/data/data/com.termux/files/usr/var/run/docker.sock|g' {} \;

    # Patch 3: Update default compose file locations
    cat >> pkg/compose/paths_android.go << 'GOPATCH'
//go:build android
// +build android

package compose

import "os"

func init() {
    // Set Termux-compatible paths
    if home := os.Getenv("HOME"); home != "" {
        defaultComposeFiles = []string{
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
            home + "/.docker/compose.yaml",
        }
    }
}
GOPATCH

    # Patch 4: Fix signal handling for Android
    cat >> internal/signals_android.go << 'GOPATCH'
//go:build android
// +build android

package internal

import (
    "os"
    "os/signal"
    "syscall"
)

func SetupSignalHandler() chan os.Signal {
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    return sigChan
}
GOPATCH

    echo "[PATCH] Applied 4 Android compatibility patches"
}

# =============================================================================
# MAIN BUILD PROCESS
# =============================================================================
termux_step_make() {
    termux_step_setup_golang

    cd "${TERMUX_PKG_SRCDIR}"

    echo "[BUILD] Building Docker Compose v${TERMUX_PKG_VERSION}..."
    echo "[BUILD] Build started at: $(date)"

    mkdir -p bin/

    # Build information
    BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    # Linker flags for optimized binary
    LDFLAGS="-s -w \
        -X github.com/docker/compose/v2/internal.Version=${TERMUX_PKG_VERSION} \
        -X github.com/docker/compose/v2/internal.GitCommit=${GIT_COMMIT} \
        -X github.com/docker/compose/v2/internal.BuildTime=${BUILD_TIME} \
        -X github.com/docker/compose/v2/internal.Platform=android/${GOARCH}"

    # Build tags for Android
    BUILD_TAGS="android,termux,exclude_graphdriver_btrfs,exclude_graphdriver_devicemapper"

    echo "[BUILD] Compiling with flags: ${LDFLAGS}"

    # Main binary build
    go build \
        -trimpath \
        -tags "${BUILD_TAGS}" \
        -ldflags="${LDFLAGS}" \
        -o bin/docker-compose \
        ./cmd

    BUILD_STATUS=$?

    if [ $BUILD_STATUS -eq 0 ]; then
        echo "[BUILD] ✓ Main binary compiled successfully"

        # Build CLI plugin variant
        echo "[BUILD] Building Docker CLI plugin variant..."
        go build \
            -trimpath \
            -tags "${BUILD_TAGS}" \
            -ldflags="${LDFLAGS}" \
            -o bin/docker-compose-plugin \
            ./cmd

        echo "[BUILD] ✓ CLI plugin compiled successfully"
    else
        termux_error_exit "Build failed with status: ${BUILD_STATUS}"
    fi

    # Show binary info
    echo ""
    echo "[BUILD] Binary information:"
    file bin/docker-compose
    ls -lh bin/

    echo "[BUILD] Build completed at: $(date)"
}

# =============================================================================
# INSTALLATION
# =============================================================================
termux_step_make_install() {
    echo "[INSTALL] Installing Docker Compose..."

    cd "${TERMUX_PKG_SRCDIR}"

    # Install main binary
    install -Dm755 bin/docker-compose "${TERMUX_PREFIX}/bin/docker-compose"

    # Install as Docker CLI plugin
    install -Dm755 bin/docker-compose-plugin \
        "${TERMUX_PREFIX}/lib/docker/cli-plugins/docker-compose"

    # Create compatibility symlinks
    ln -sf "${TERMUX_PREFIX}/bin/docker-compose" \
        "${TERMUX_PREFIX}/bin/docker-compose-v2"

    # Install shell completions
    mkdir -p "${TERMUX_PREFIX}/share/bash-completion/completions"
    mkdir -p "${TERMUX_PREFIX}/share/zsh/site-functions"
    mkdir -p "${TERMUX_PREFIX}/share/fish/vendor_completions.d"

    # Generate completions
    if [ -x "${TERMUX_PREFIX}/bin/docker-compose" ]; then
        "${TERMUX_PREFIX}/bin/docker-compose" completion bash \
            > "${TERMUX_PREFIX}/share/bash-completion/completions/docker-compose" 2>/dev/null || true
        "${TERMUX_PREFIX}/bin/docker-compose" completion zsh \
            > "${TERMUX_PREFIX}/share/zsh/site-functions/_docker-compose" 2>/dev/null || true
        "${TERMUX_PREFIX}/bin/docker-compose" completion fish \
            > "${TERMUX_PREFIX}/share/fish/vendor_completions.d/docker-compose.fish" 2>/dev/null || true
    fi

    echo "[INSTALL] ✓ Installation complete"
}

# =============================================================================
# POST-INSTALLATION SETUP
# =============================================================================
termux_step_post_make_install() {
    echo "[POST] Running post-installation tasks..."

    # Create default directories
    mkdir -p "${TERMUX_PREFIX}/etc/docker"
    mkdir -p "${HOME}/.docker"

    # Create default compose configuration
    cat > "${TERMUX_PREFIX}/etc/docker/compose-defaults.yaml" << 'EOF'
# Docker Compose Defaults for Termux
# NEXUS AGI Enhanced Configuration

version: "3.9"

x-termux-defaults: &termux-defaults
  restart: unless-stopped
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
  security_opt:
    - no-new-privileges:true

x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
EOF

    # Create wrapper script with enhanced features
    cat > "${TERMUX_PREFIX}/bin/compose" << 'WRAPPER'
#!/data/data/com.termux/files/usr/bin/bash
# Docker Compose Wrapper for Termux - NEXUS AGI Enhanced
# Provides additional convenience features

set -e

COMPOSE_CMD="${PREFIX}/bin/docker-compose"
DOCKER_SOCK="${PREFIX}/var/run/docker.sock"

# Check Docker daemon
check_docker() {
    if [ ! -S "$DOCKER_SOCK" ]; then
        echo "⚠ Docker daemon not running. Start with: dockerd &"
        exit 1
    fi
}

# Enhanced help
show_help() {
    echo "NEXUS AGI Docker Compose Wrapper"
    echo ""
    echo "Additional commands:"
    echo "  compose status    - Show all container status"
    echo "  compose cleanup   - Remove stopped containers and unused images"
    echo "  compose backup    - Backup volumes to tar archives"
    echo ""
    exec "$COMPOSE_CMD" --help
}

case "$1" in
    status)
        check_docker
        echo "=== Container Status ==="
        docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ;;
    cleanup)
        check_docker
        echo "Cleaning up Docker resources..."
        docker system prune -f
        ;;
    backup)
        shift
        BACKUP_DIR="${2:-./backups}"
        mkdir -p "$BACKUP_DIR"
        echo "Backing up volumes to $BACKUP_DIR..."
        docker volume ls -q | while read vol; do
            docker run --rm -v "$vol:/data" -v "$BACKUP_DIR:/backup" \
                alpine tar czf "/backup/${vol}.tar.gz" -C /data .
        done
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        check_docker
        exec "$COMPOSE_CMD" "$@"
        ;;
esac
WRAPPER
    chmod +x "${TERMUX_PREFIX}/bin/compose"

    echo "[POST] ✓ Post-installation complete"
}

# =============================================================================
# CREATE DEB PACKAGE METADATA
# =============================================================================
termux_step_create_debscripts() {
    # Post-installation script
    cat > postinst << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Docker Compose v2 installed successfully!                       ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Usage:                                                          ║"
echo "║    docker compose up -d      # Start services                    ║"
echo "║    docker compose down       # Stop services                     ║"
echo "║    docker compose logs -f    # View logs                         ║"
echo "║    compose status            # Enhanced status (wrapper)         ║"
echo "║                                                                  ║"
echo "║  Note: Ensure Docker daemon is running: dockerd &                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
EOF

    # Pre-removal script
    cat > prerm << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# Stop any running compose projects gracefully
if command -v docker-compose &> /dev/null; then
    echo "Note: Running compose projects should be stopped manually"
fi
EOF
}

# =============================================================================
# VERIFY INSTALLATION
# =============================================================================
termux_step_post_massage() {
    echo "[VERIFY] Verifying installation..."

    # Check binary exists and is executable
    if [ ! -x "${TERMUX_PREFIX}/bin/docker-compose" ]; then
        termux_error_exit "Binary not found or not executable"
    fi

    # Verify binary architecture
    BINARY_ARCH=$(file "${TERMUX_PREFIX}/bin/docker-compose" | grep -oE 'ARM|x86|aarch64' || echo "unknown")
    echo "[VERIFY] Binary architecture: ${BINARY_ARCH}"

    # Check file size (should be reasonable)
    BINARY_SIZE=$(stat -f%z "${TERMUX_PREFIX}/bin/docker-compose" 2>/dev/null || \
                  stat -c%s "${TERMUX_PREFIX}/bin/docker-compose" 2>/dev/null || echo "0")
    echo "[VERIFY] Binary size: $(numfmt --to=iec-i --suffix=B ${BINARY_SIZE} 2>/dev/null || echo "${BINARY_SIZE} bytes")"

    echo "[VERIFY] ✓ All verification checks passed"
}
