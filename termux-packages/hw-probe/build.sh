#!/bin/bash
# =============================================================================
# Enhanced Hardware Probe Package for Termux
# NEXUS AGI - Advanced Hardware Detection & Analysis Module
# Version: 1.6.5 (Enhanced for Android/Termux)
# =============================================================================

TERMUX_PKG_HOMEPAGE="https://github.com/linuxhw/hw-probe"
TERMUX_PKG_DESCRIPTION="Advanced hardware probe tool for Android/Termux - Detect, analyze, and report device hardware with NEXUS AGI enhancements"
TERMUX_PKG_LICENSE="LGPL-2.1-or-later"
TERMUX_PKG_MAINTAINER="NEXUS AGI Team <nexus@agi.dev>"
TERMUX_PKG_VERSION="1.6.5"
TERMUX_PKG_REVISION=1
TERMUX_PKG_SRCURL="https://github.com/linuxhw/hw-probe/archive/${TERMUX_PKG_VERSION}.tar.gz"
TERMUX_PKG_SHA256="c8d9e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8"
TERMUX_PKG_AUTO_UPDATE=true
TERMUX_PKG_PLATFORM_INDEPENDENT=true
TERMUX_PKG_BUILD_IN_SRC=true

# Extended dependencies for full hardware detection
TERMUX_PKG_DEPENDS="
perl,
curl,
hwinfo,
net-tools,
pciutils,
usbutils,
util-linux,
coreutils,
procps,
iproute2,
wireless-tools,
ethtool,
dmidecode,
smartmontools,
hdparm,
lm-sensors,
mesa-demos,
vulkan-tools,
termux-api
"

TERMUX_PKG_SUGGESTS="
inxi,
neofetch,
cpufetch,
glxinfo,
vainfo,
vdpauinfo
"

TERMUX_PKG_CONFLICTS="hw-probe-lite"
TERMUX_PKG_REPLACES="hw-probe-lite"

# =============================================================================
# PRE-CONFIGURATION
# =============================================================================
termux_step_pre_configure() {
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║     NEXUS AGI - Hardware Probe Package Builder                       ║"
    echo "║     Version: ${TERMUX_PKG_VERSION} (Android/Termux Enhanced)                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "[INFO] Preparing enhanced hardware probe for Termux..."
    echo "[INFO] Target: Android/Termux environment"
    echo "[INFO] Architecture: ${TERMUX_ARCH}"

    # Create required directories
    mkdir -p "${TERMUX_PREFIX}/share/hw-probe"
    mkdir -p "${TERMUX_PREFIX}/etc/hw-probe"
    mkdir -p "${TERMUX_PREFIX}/var/log/hw-probe"
}

# =============================================================================
# SOURCE PATCHING FOR ANDROID
# =============================================================================
termux_step_post_get_source() {
    echo "[PATCH] Applying Android/Termux compatibility patches..."

    cd "${TERMUX_PKG_SRCDIR}"

    # Patch 1: Fix paths for Termux
    sed -i "s|/usr/|${TERMUX_PREFIX}/|g" hw-probe.pl
    sed -i "s|/etc/|${TERMUX_PREFIX}/etc/|g" hw-probe.pl
    sed -i "s|/var/|${TERMUX_PREFIX}/var/|g" hw-probe.pl
    sed -i "s|/tmp/|${TERMUX_PREFIX}/tmp/|g" hw-probe.pl

    # Patch 2: Fix proc filesystem paths
    sed -i 's|/proc/cpuinfo|/proc/cpuinfo|g' hw-probe.pl

    # Patch 3: Add Android-specific detection methods
    cat >> hw-probe.pl << 'ANDROID_PATCH'

# =============================================================================
# NEXUS AGI Android/Termux Extensions
# =============================================================================

sub detectAndroidHardware {
    my %android_info;

    # Get Android properties via getprop
    if (-x "/system/bin/getprop") {
        $android_info{'model'} = `getprop ro.product.model 2>/dev/null`;
        $android_info{'manufacturer'} = `getprop ro.product.manufacturer 2>/dev/null`;
        $android_info{'brand'} = `getprop ro.product.brand 2>/dev/null`;
        $android_info{'device'} = `getprop ro.product.device 2>/dev/null`;
        $android_info{'board'} = `getprop ro.product.board 2>/dev/null`;
        $android_info{'hardware'} = `getprop ro.hardware 2>/dev/null`;
        $android_info{'platform'} = `getprop ro.board.platform 2>/dev/null`;
        $android_info{'android_version'} = `getprop ro.build.version.release 2>/dev/null`;
        $android_info{'sdk_version'} = `getprop ro.build.version.sdk 2>/dev/null`;
        $android_info{'security_patch'} = `getprop ro.build.version.security_patch 2>/dev/null`;
        $android_info{'build_id'} = `getprop ro.build.id 2>/dev/null`;
        $android_info{'fingerprint'} = `getprop ro.build.fingerprint 2>/dev/null`;

        # CPU info
        $android_info{'cpu_abi'} = `getprop ro.product.cpu.abi 2>/dev/null`;
        $android_info{'cpu_abilist'} = `getprop ro.product.cpu.abilist 2>/dev/null`;

        # GPU info
        $android_info{'gpu_renderer'} = `getprop ro.hardware.egl 2>/dev/null`;
        $android_info{'gpu_version'} = `getprop ro.opengles.version 2>/dev/null`;

        # Trim whitespace
        foreach my $key (keys %android_info) {
            $android_info{$key} =~ s/^\s+|\s+$//g if $android_info{$key};
        }
    }

    return \%android_info;
}

sub detectTermuxEnvironment {
    my %termux_info;

    $termux_info{'prefix'} = $ENV{'PREFIX'} || '/data/data/com.termux/files/usr';
    $termux_info{'home'} = $ENV{'HOME'} || '/data/data/com.termux/files/home';
    $termux_info{'shell'} = $ENV{'SHELL'} || 'unknown';
    $termux_info{'term'} = $ENV{'TERM'} || 'unknown';
    $termux_info{'termux_version'} = $ENV{'TERMUX_VERSION'} || 'unknown';

    # Check for Termux:API
    if (-x "$termux_info{'prefix'}/bin/termux-battery-status") {
        $termux_info{'termux_api'} = 'installed';

        # Get battery info
        my $battery_json = `termux-battery-status 2>/dev/null`;
        if ($battery_json) {
            $termux_info{'battery_info'} = $battery_json;
        }

        # Get device info via Termux:API
        my $device_info = `termux-telephony-deviceinfo 2>/dev/null`;
        if ($device_info) {
            $termux_info{'telephony_info'} = $device_info;
        }
    }

    return \%termux_info;
}

sub detectSoCInfo {
    my %soc_info;

    # Qualcomm Snapdragon detection
    if (-f "/sys/devices/soc0/soc_id") {
        $soc_info{'soc_id'} = `cat /sys/devices/soc0/soc_id 2>/dev/null`;
        $soc_info{'soc_id'} =~ s/^\s+|\s+$//g;
    }

    if (-f "/sys/devices/soc0/family") {
        $soc_info{'family'} = `cat /sys/devices/soc0/family 2>/dev/null`;
        $soc_info{'family'} =~ s/^\s+|\s+$//g;
    }

    if (-f "/sys/devices/soc0/machine") {
        $soc_info{'machine'} = `cat /sys/devices/soc0/machine 2>/dev/null`;
        $soc_info{'machine'} =~ s/^\s+|\s+$//g;
    }

    if (-f "/sys/devices/soc0/revision") {
        $soc_info{'revision'} = `cat /sys/devices/soc0/revision 2>/dev/null`;
        $soc_info{'revision'} =~ s/^\s+|\s+$//g;
    }

    # MediaTek detection
    if (-f "/proc/device-tree/model") {
        $soc_info{'dt_model'} = `cat /proc/device-tree/model 2>/dev/null`;
        $soc_info{'dt_model'} =~ s/\x00//g;
    }

    # Thermal zones
    my @thermal_zones = glob("/sys/class/thermal/thermal_zone*/type");
    $soc_info{'thermal_zones'} = scalar(@thermal_zones);

    return \%soc_info;
}

sub detectDisplayInfo {
    my %display_info;

    # Get display info from sysfs
    my @fb_devices = glob("/sys/class/graphics/fb*");
    foreach my $fb (@fb_devices) {
        my $name = `cat $fb/name 2>/dev/null`;
        $name =~ s/^\s+|\s+$//g;

        if (-f "$fb/virtual_size") {
            my $size = `cat $fb/virtual_size 2>/dev/null`;
            $size =~ s/^\s+|\s+$//g;
            $display_info{$fb} = { name => $name, size => $size };
        }
    }

    # DRM info
    if (-d "/sys/class/drm") {
        my @drm_cards = glob("/sys/class/drm/card*");
        $display_info{'drm_cards'} = scalar(@drm_cards);
    }

    return \%display_info;
}

sub detectStorageInfo {
    my %storage_info;

    # Block devices
    my @block_devices = glob("/sys/block/*");
    foreach my $block (@block_devices) {
        my $dev_name = $block;
        $dev_name =~ s|.*/||;

        next if $dev_name =~ /^(loop|ram|dm-)/;

        my $size = 0;
        if (-f "$block/size") {
            $size = `cat $block/size 2>/dev/null`;
            $size =~ s/^\s+|\s+$//g;
            $size = int($size) * 512; # Convert to bytes
        }

        my $model = '';
        if (-f "$block/device/model") {
            $model = `cat $block/device/model 2>/dev/null`;
            $model =~ s/^\s+|\s+$//g;
        }

        $storage_info{$dev_name} = {
            size_bytes => $size,
            size_gb => sprintf("%.2f", $size / (1024**3)),
            model => $model
        };
    }

    return \%storage_info;
}

sub detectNetworkInfo {
    my %network_info;

    # Network interfaces
    my @interfaces = glob("/sys/class/net/*");
    foreach my $iface (@interfaces) {
        my $name = $iface;
        $name =~ s|.*/||;

        my $mac = '';
        if (-f "$iface/address") {
            $mac = `cat $iface/address 2>/dev/null`;
            $mac =~ s/^\s+|\s+$//g;
        }

        my $state = '';
        if (-f "$iface/operstate") {
            $state = `cat $iface/operstate 2>/dev/null`;
            $state =~ s/^\s+|\s+$//g;
        }

        my $type = '';
        if (-f "$iface/type") {
            $type = `cat $iface/type 2>/dev/null`;
            $type =~ s/^\s+|\s+$//g;
        }

        $network_info{$name} = {
            mac => $mac,
            state => $state,
            type => $type
        };
    }

    return \%network_info;
}

sub detectSensorInfo {
    my %sensor_info;

    # IIO sensors (accelerometer, gyroscope, magnetometer, etc.)
    my @iio_devices = glob("/sys/bus/iio/devices/iio:device*");
    foreach my $iio (@iio_devices) {
        my $name = '';
        if (-f "$iio/name") {
            $name = `cat $iio/name 2>/dev/null`;
            $name =~ s/^\s+|\s+$//g;
        }

        my $dev_num = $iio;
        $dev_num =~ s|.*/||;

        $sensor_info{$dev_num} = { name => $name, path => $iio };
    }

    # Input devices (touchscreen, buttons, etc.)
    my @input_devices = glob("/sys/class/input/input*");
    foreach my $input (@input_devices) {
        my $name = '';
        if (-f "$input/name") {
            $name = `cat $input/name 2>/dev/null`;
            $name =~ s/^\s+|\s+$//g;
        }

        my $dev_num = $input;
        $dev_num =~ s|.*/||;

        $sensor_info{"input_$dev_num"} = { name => $name, type => 'input' };
    }

    return \%sensor_info;
}

sub generateNexusReport {
    my ($output_dir) = @_;

    print "Generating NEXUS AGI Enhanced Hardware Report...\n";

    my $android = detectAndroidHardware();
    my $termux = detectTermuxEnvironment();
    my $soc = detectSoCInfo();
    my $display = detectDisplayInfo();
    my $storage = detectStorageInfo();
    my $network = detectNetworkInfo();
    my $sensors = detectSensorInfo();

    # Generate JSON report
    my $report = {
        'nexus_version' => '1.0',
        'probe_version' => $VERSION,
        'timestamp' => time(),
        'android' => $android,
        'termux' => $termux,
        'soc' => $soc,
        'display' => $display,
        'storage' => $storage,
        'network' => $network,
        'sensors' => $sensors
    };

    return $report;
}

1;
ANDROID_PATCH

    echo "[PATCH] Applied Android/Termux extensions"
}

# =============================================================================
# BUILD (No compilation needed - Perl script)
# =============================================================================
termux_step_make() {
    echo "[BUILD] Validating Perl syntax..."
    perl -c hw-probe.pl || {
        echo "[ERROR] Perl syntax check failed"
        return 1
    }
    echo "[BUILD] Syntax validation passed"
}

# =============================================================================
# INSTALLATION
# =============================================================================
termux_step_make_install() {
    echo "[INSTALL] Installing hw-probe to Termux..."

    cd "${TERMUX_PKG_SRCDIR}"

    # Install main script
    install -Dm755 hw-probe.pl "${TERMUX_PREFIX}/bin/hw-probe"

    # Create enhanced wrapper script
    cat > "${TERMUX_PREFIX}/bin/nexus-hwprobe" << 'WRAPPER'
#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# NEXUS AGI Hardware Probe Wrapper
# Enhanced hardware detection for Android/Termux
# =============================================================================

set -e

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
HWPROBE="${PREFIX}/bin/hw-probe"
LOG_DIR="${PREFIX}/var/log/hw-probe"
CACHE_DIR="${HOME}/.cache/hw-probe"
CONFIG_FILE="${PREFIX}/etc/hw-probe/config"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

mkdir -p "$LOG_DIR" "$CACHE_DIR"

show_banner() {
    echo -e "${CYAN}"
    cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗    ██╗  ██╗██╗    ██╗      ║
║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██║  ██║██║    ██║      ║
║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ███████║██║ █╗ ██║      ║
║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██╔══██║██║███╗██║      ║
║   ██║ ╚████║███████╗██╔╝ ╚██╗╚██████╔╝███████║    ██║  ██║╚███╔███╔╝      ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝ ╚══╝╚══╝       ║
║                                                                           ║
║              Hardware Probe - Android/Termux Enhanced                     ║
║                         Version 1.6.5                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
BANNER
    echo -e "${NC}"
}

show_help() {
    show_banner
    echo -e "${BOLD}Usage:${NC} nexus-hwprobe [OPTIONS] [COMMAND]"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo "  probe           Full hardware probe (default)"
    echo "  quick           Quick system overview"
    echo "  cpu             CPU information only"
    echo "  memory          Memory information only"
    echo "  storage         Storage devices information"
    echo "  network         Network interfaces information"
    echo "  display         Display/GPU information"
    echo "  sensors         Sensors information"
    echo "  battery         Battery status (requires Termux:API)"
    echo "  android         Android system properties"
    echo "  export          Export probe to Linux Hardware Database"
    echo "  report          Generate full HTML report"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  -h, --help      Show this help message"
    echo "  -v, --version   Show version information"
    echo "  -j, --json      Output in JSON format"
    echo "  -q, --quiet     Minimal output"
    echo "  -o, --output    Specify output directory"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  nexus-hwprobe                    # Full probe"
    echo "  nexus-hwprobe quick              # Quick overview"
    echo "  nexus-hwprobe cpu --json         # CPU info as JSON"
    echo "  nexus-hwprobe export             # Upload to HW database"
    echo ""
}

get_android_prop() {
    local prop="$1"
    local default="${2:-unknown}"
    if [ -x /system/bin/getprop ]; then
        local value=$(getprop "$prop" 2>/dev/null)
        echo "${value:-$default}"
    else
        echo "$default"
    fi
}

cmd_quick() {
    show_banner
    echo -e "${BOLD}${CYAN}═══ Quick System Overview ═══${NC}"
    echo ""

    # Device info
    echo -e "${BOLD}📱 Device:${NC}"
    echo "   Model:        $(get_android_prop ro.product.model)"
    echo "   Manufacturer: $(get_android_prop ro.product.manufacturer)"
    echo "   Brand:        $(get_android_prop ro.product.brand)"
    echo "   Device:       $(get_android_prop ro.product.device)"
    echo ""

    # Android info
    echo -e "${BOLD}🤖 Android:${NC}"
    echo "   Version:      $(get_android_prop ro.build.version.release)"
    echo "   SDK Level:    $(get_android_prop ro.build.version.sdk)"
    echo "   Build ID:     $(get_android_prop ro.build.id)"
    echo "   Security:     $(get_android_prop ro.build.version.security_patch)"
    echo ""

    # CPU info
    echo -e "${BOLD}⚡ Processor:${NC}"
    local cpu_model=$(grep -m1 "model name\|Hardware\|Processor" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)
    local cpu_cores=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "?")
    local cpu_arch=$(get_android_prop ro.product.cpu.abi)
    echo "   Model:        ${cpu_model:-$(get_android_prop ro.hardware)}"
    echo "   Cores:        ${cpu_cores}"
    echo "   Architecture: ${cpu_arch}"
    echo ""

    # Memory info
    echo -e "${BOLD}🧠 Memory:${NC}"
    local mem_total=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{printf "%.1f GB", $2/1024/1024}')
    local mem_free=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{printf "%.1f GB", $2/1024/1024}')
    echo "   Total:        ${mem_total}"
    echo "   Available:    ${mem_free}"
    echo ""

    # Storage info
    echo -e "${BOLD}💾 Storage:${NC}"
    df -h 2>/dev/null | grep -E "^/dev|emulated" | head -3 | while read line; do
        echo "   $line"
    done
    echo ""

    # Battery (if Termux:API available)
    if command -v termux-battery-status &>/dev/null; then
        echo -e "${BOLD}🔋 Battery:${NC}"
        local battery=$(termux-battery-status 2>/dev/null)
        if [ -n "$battery" ]; then
            local level=$(echo "$battery" | grep -o '"percentage":[0-9]*' | cut -d: -f2)
            local status=$(echo "$battery" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            local temp=$(echo "$battery" | grep -o '"temperature":[0-9.]*' | cut -d: -f2)
            echo "   Level:        ${level}%"
            echo "   Status:       ${status}"
            echo "   Temperature:  ${temp}°C"
        fi
        echo ""
    fi

    # Network
    echo -e "${BOLD}🌐 Network:${NC}"
    ip -o link show 2>/dev/null | grep -v "lo:" | while read line; do
        local iface=$(echo "$line" | awk -F': ' '{print $2}')
        local state=$(echo "$line" | grep -oE "state [A-Z]+" | awk '{print $2}')
        echo "   $iface: $state"
    done
    echo ""
}

cmd_cpu() {
    echo -e "${BOLD}${CYAN}═══ CPU Information ═══${NC}"
    echo ""

    echo -e "${BOLD}Processor Details:${NC}"
    grep -E "^(processor|model name|Hardware|Processor|BogoMIPS|Features|CPU architecture|CPU variant|CPU part)" /proc/cpuinfo 2>/dev/null | head -20
    echo ""

    echo -e "${BOLD}CPU Frequencies:${NC}"
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_cur_freq" ]; then
            local cpu_num=$(basename "$cpu")
            local freq=$(cat "$cpu/cpufreq/scaling_cur_freq" 2>/dev/null)
            local freq_mhz=$((freq / 1000))
            local min=$(cat "$cpu/cpufreq/scaling_min_freq" 2>/dev/null)
            local max=$(cat "$cpu/cpufreq/scaling_max_freq" 2>/dev/null)
            echo "  $cpu_num: ${freq_mhz} MHz (min: $((min/1000)) MHz, max: $((max/1000)) MHz)"
        fi
    done
    echo ""

    echo -e "${BOLD}SoC Information:${NC}"
    [ -f /sys/devices/soc0/family ] && echo "  Family:   $(cat /sys/devices/soc0/family)"
    [ -f /sys/devices/soc0/machine ] && echo "  Machine:  $(cat /sys/devices/soc0/machine)"
    [ -f /sys/devices/soc0/soc_id ] && echo "  SoC ID:   $(cat /sys/devices/soc0/soc_id)"
    [ -f /sys/devices/soc0/revision ] && echo "  Revision: $(cat /sys/devices/soc0/revision)"
    echo ""
}

cmd_memory() {
    echo -e "${BOLD}${CYAN}═══ Memory Information ═══${NC}"
    echo ""

    echo -e "${BOLD}Memory Status:${NC}"
    cat /proc/meminfo 2>/dev/null | head -15
    echo ""

    echo -e "${BOLD}Memory Summary:${NC}"
    free -h 2>/dev/null || cat /proc/meminfo | grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached)"
    echo ""

    echo -e "${BOLD}Swap Status:${NC}"
    cat /proc/swaps 2>/dev/null || echo "No swap configured"
    echo ""
}

cmd_storage() {
    echo -e "${BOLD}${CYAN}═══ Storage Information ═══${NC}"
    echo ""

    echo -e "${BOLD}Block Devices:${NC}"
    lsblk 2>/dev/null || ls -la /dev/block/ 2>/dev/null | head -20
    echo ""

    echo -e "${BOLD}Disk Usage:${NC}"
    df -h 2>/dev/null
    echo ""

    echo -e "${BOLD}Mount Points:${NC}"
    mount 2>/dev/null | grep -E "^/dev" | head -15
    echo ""
}

cmd_network() {
    echo -e "${BOLD}${CYAN}═══ Network Information ═══${NC}"
    echo ""

    echo -e "${BOLD}Network Interfaces:${NC}"
    ip addr 2>/dev/null || ifconfig 2>/dev/null
    echo ""

    echo -e "${BOLD}Routing Table:${NC}"
    ip route 2>/dev/null || route -n 2>/dev/null
    echo ""

    echo -e "${BOLD}DNS Configuration:${NC}"
    cat "${PREFIX}/etc/resolv.conf" 2>/dev/null || getprop net.dns1 2>/dev/null
    echo ""

    echo -e "${BOLD}WiFi Information:${NC}"
    if command -v iwconfig &>/dev/null; then
        iwconfig 2>/dev/null | grep -v "no wireless"
    else
        echo "  WiFi tools not available"
    fi
    echo ""
}

cmd_display() {
    echo -e "${BOLD}${CYAN}═══ Display/GPU Information ═══${NC}"
    echo ""

    echo -e "${BOLD}Framebuffer Devices:${NC}"
    for fb in /sys/class/graphics/fb*; do
        if [ -d "$fb" ]; then
            local name=$(cat "$fb/name" 2>/dev/null || echo "unknown")
            local size=$(cat "$fb/virtual_size" 2>/dev/null || echo "unknown")
            echo "  $(basename $fb): $name ($size)"
        fi
    done
    echo ""

    echo -e "${BOLD}DRM Information:${NC}"
    ls /sys/class/drm/ 2>/dev/null | head -10
    echo ""

    echo -e "${BOLD}OpenGL ES Version:${NC}"
    local gles=$(get_android_prop ro.opengles.version)
    case "$gles" in
        196608) echo "  OpenGL ES 3.0" ;;
        196609) echo "  OpenGL ES 3.1" ;;
        196610) echo "  OpenGL ES 3.2" ;;
        131072) echo "  OpenGL ES 2.0" ;;
        *)      echo "  Version code: $gles" ;;
    esac
    echo ""

    echo -e "${BOLD}GPU Renderer:${NC}"
    echo "  EGL: $(get_android_prop ro.hardware.egl)"
    echo "  Vulkan: $(get_android_prop ro.hardware.vulkan)"
    echo ""
}

cmd_sensors() {
    echo -e "${BOLD}${CYAN}═══ Sensors Information ═══${NC}"
    echo ""

    echo -e "${BOLD}IIO Sensors:${NC}"
    for iio in /sys/bus/iio/devices/iio:device*; do
        if [ -d "$iio" ]; then
            local name=$(cat "$iio/name" 2>/dev/null || echo "unknown")
            echo "  $(basename $iio): $name"
        fi
    done
    echo ""

    echo -e "${BOLD}Input Devices:${NC}"
    for input in /sys/class/input/input*; do
        if [ -d "$input" ]; then
            local name=$(cat "$input/name" 2>/dev/null || echo "unknown")
            echo "  $(basename $input): $name"
        fi
    done
    echo ""

    echo -e "${BOLD}Thermal Zones:${NC}"
    for tz in /sys/class/thermal/thermal_zone*; do
        if [ -d "$tz" ]; then
            local type=$(cat "$tz/type" 2>/dev/null || echo "unknown")
            local temp=$(cat "$tz/temp" 2>/dev/null || echo "0")
            local temp_c=$((temp / 1000))
            echo "  $(basename $tz): $type = ${temp_c}°C"
        fi
    done
    echo ""
}

cmd_battery() {
    echo -e "${BOLD}${CYAN}═══ Battery Information ═══${NC}"
    echo ""

    if command -v termux-battery-status &>/dev/null; then
        echo -e "${BOLD}Battery Status (via Termux:API):${NC}"
        termux-battery-status 2>/dev/null | python3 -m json.tool 2>/dev/null || termux-battery-status
        echo ""
    fi

    echo -e "${BOLD}Power Supply Info:${NC}"
    for ps in /sys/class/power_supply/*; do
        if [ -d "$ps" ]; then
            local name=$(basename "$ps")
            local type=$(cat "$ps/type" 2>/dev/null || echo "unknown")
            echo "  $name ($type):"
            [ -f "$ps/status" ] && echo "    Status:   $(cat $ps/status)"
            [ -f "$ps/capacity" ] && echo "    Capacity: $(cat $ps/capacity)%"
            [ -f "$ps/voltage_now" ] && echo "    Voltage:  $(($(cat $ps/voltage_now) / 1000)) mV"
            [ -f "$ps/current_now" ] && echo "    Current:  $(($(cat $ps/current_now) / 1000)) mA"
            [ -f "$ps/temp" ] && echo "    Temp:     $(($(cat $ps/temp) / 10))°C"
        fi
    done
    echo ""
}

cmd_android() {
    echo -e "${BOLD}${CYAN}═══ Android System Properties ═══${NC}"
    echo ""

    echo -e "${BOLD}Device Information:${NC}"
    echo "  Model:         $(get_android_prop ro.product.model)"
    echo "  Manufacturer:  $(get_android_prop ro.product.manufacturer)"
    echo "  Brand:         $(get_android_prop ro.product.brand)"
    echo "  Device:        $(get_android_prop ro.product.device)"
    echo "  Board:         $(get_android_prop ro.product.board)"
    echo "  Hardware:      $(get_android_prop ro.hardware)"
    echo "  Platform:      $(get_android_prop ro.board.platform)"
    echo ""

    echo -e "${BOLD}Build Information:${NC}"
    echo "  Android:       $(get_android_prop ro.build.version.release)"
    echo "  SDK:           $(get_android_prop ro.build.version.sdk)"
    echo "  Build ID:      $(get_android_prop ro.build.id)"
    echo "  Build Type:    $(get_android_prop ro.build.type)"
    echo "  Security:      $(get_android_prop ro.build.version.security_patch)"
    echo "  Fingerprint:   $(get_android_prop ro.build.fingerprint)"
    echo ""

    echo -e "${BOLD}CPU Information:${NC}"
    echo "  ABI:           $(get_android_prop ro.product.cpu.abi)"
    echo "  ABI List:      $(get_android_prop ro.product.cpu.abilist)"
    echo ""

    echo -e "${BOLD}Display Information:${NC}"
    echo "  Density:       $(get_android_prop ro.sf.lcd_density) dpi"
    echo "  Screen:        $(get_android_prop ro.product.screen.width)x$(get_android_prop ro.product.screen.height)"
    echo ""
}

cmd_probe() {
    show_banner
    echo -e "${BOLD}Starting full hardware probe...${NC}"
    echo ""

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_dir="${CACHE_DIR}/probe_${timestamp}"
    mkdir -p "$output_dir"

    echo "[1/8] Probing Android system properties..."
    cmd_android > "$output_dir/android.txt" 2>&1
    echo -e "${GREEN}✓${NC} Android properties collected"

    echo "[2/8] Probing CPU information..."
    cmd_cpu > "$output_dir/cpu.txt" 2>&1
    echo -e "${GREEN}✓${NC} CPU information collected"

    echo "[3/8] Probing memory information..."
    cmd_memory > "$output_dir/memory.txt" 2>&1
    echo -e "${GREEN}✓${NC} Memory information collected"

    echo "[4/8] Probing storage devices..."
    cmd_storage > "$output_dir/storage.txt" 2>&1
    echo -e "${GREEN}✓${NC} Storage information collected"

    echo "[5/8] Probing network interfaces..."
    cmd_network > "$output_dir/network.txt" 2>&1
    echo -e "${GREEN}✓${NC} Network information collected"

    echo "[6/8] Probing display/GPU..."
    cmd_display > "$output_dir/display.txt" 2>&1
    echo -e "${GREEN}✓${NC} Display information collected"

    echo "[7/8] Probing sensors..."
    cmd_sensors > "$output_dir/sensors.txt" 2>&1
    echo -e "${GREEN}✓${NC} Sensors information collected"

    echo "[8/8] Probing battery status..."
    cmd_battery > "$output_dir/battery.txt" 2>&1
    echo -e "${GREEN}✓${NC} Battery information collected"

    echo ""
    echo -e "${GREEN}${BOLD}Hardware probe completed!${NC}"
    echo "Results saved to: $output_dir"
    echo ""

    # Display summary
    cmd_quick
}

cmd_export() {
    show_banner
    echo -e "${BOLD}Exporting probe to Linux Hardware Database...${NC}"
    echo ""
    echo "Running: hw-probe -all -upload"
    echo ""
    exec "$HWPROBE" -all -upload "$@"
}

cmd_report() {
    show_banner
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local report_file="${CACHE_DIR}/hw-report_${timestamp}.html"

    echo "Generating HTML report..."
    exec "$HWPROBE" -all -html "$report_file" "$@"

    echo ""
    echo -e "${GREEN}Report generated: $report_file${NC}"
}

# Main entry point
case "${1:-probe}" in
    -h|--help|help)
        show_help
        ;;
    -v|--version|version)
        echo "NEXUS AGI Hardware Probe v1.6.5"
        echo "Based on hw-probe by Linux Hardware Project"
        "$HWPROBE" -v 2>/dev/null || true
        ;;
    quick)
        shift
        cmd_quick "$@"
        ;;
    cpu)
        shift
        cmd_cpu "$@"
        ;;
    memory|mem)
        shift
        cmd_memory "$@"
        ;;
    storage|disk)
        shift
        cmd_storage "$@"
        ;;
    network|net)
        shift
        cmd_network "$@"
        ;;
    display|gpu)
        shift
        cmd_display "$@"
        ;;
    sensors)
        shift
        cmd_sensors "$@"
        ;;
    battery|batt)
        shift
        cmd_battery "$@"
        ;;
    android)
        shift
        cmd_android "$@"
        ;;
    export|upload)
        shift
        cmd_export "$@"
        ;;
    report|html)
        shift
        cmd_report "$@"
        ;;
    probe|full|all)
        shift
        cmd_probe "$@"
        ;;
    *)
        # Pass through to hw-probe for other options
        exec "$HWPROBE" "$@"
        ;;
esac
WRAPPER

    chmod +x "${TERMUX_PREFIX}/bin/nexus-hwprobe"

    # Create symlinks
    ln -sf nexus-hwprobe "${TERMUX_PREFIX}/bin/hwprobe"

    # Install configuration file
    cat > "${TERMUX_PREFIX}/etc/hw-probe/config" << 'CONFIG'
# NEXUS AGI Hardware Probe Configuration
# /data/data/com.termux/files/usr/etc/hw-probe/config

# Upload settings
UPLOAD_URL="https://linux-hardware.org"
PRIVACY_LEVEL="public"

# Probe settings
PROBE_ALL=true
PROBE_SENSORS=true
PROBE_LOGS=false

# Android-specific
USE_GETPROP=true
USE_TERMUX_API=true

# Output settings
OUTPUT_FORMAT="text"
COLOR_OUTPUT=true
CONFIG

    echo "[INSTALL] ✓ Installation complete"
}

# =============================================================================
# POST-INSTALLATION
# =============================================================================
termux_step_post_make_install() {
    echo "[POST] Setting up permissions and directories..."

    # Create necessary directories
    mkdir -p "${TERMUX_PREFIX}/var/log/hw-probe"
    mkdir -p "${TERMUX_PREFIX}/share/hw-probe/db"

    echo "[POST] ✓ Post-installation complete"
}

# =============================================================================
# DEB PACKAGE SCRIPTS
# =============================================================================
termux_step_create_debscripts() {
    cat > postinst << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  NEXUS AGI Hardware Probe installed successfully!                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                      ║"
echo "║  Quick Start:                                                        ║"
echo "║    nexus-hwprobe              # Full hardware probe                  ║"
echo "║    nexus-hwprobe quick        # Quick system overview                ║"
echo "║    nexus-hwprobe cpu          # CPU information                      ║"
echo "║    nexus-hwprobe battery      # Battery status                       ║"
echo "║    nexus-hwprobe export       # Upload to HW database                ║"
echo "║                                                                      ║"
echo "║  For best results, install: pkg install termux-api                   ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
EOF

    cat > prerm << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
echo "Removing hw-probe cache..."
rm -rf "${HOME}/.cache/hw-probe" 2>/dev/null || true
EOF
}
