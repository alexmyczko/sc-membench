# sc-membench Makefile
#
# Portable build system for Linux, macOS, and BSD
#
# Build options:
#   make           - Auto-detect features and build optimal version
#   make basic     - Build minimal version (no optional dependencies)
#   make hwloc     - Build with hwloc (portable cache detection)
#   make numa      - Build with NUMA support (Linux only)
#   make full      - Build with all available features
#   make all       - Build all versions
#   make clean     - Remove built files
#   make test      - Quick test run
#
# The default 'make' target automatically detects:
#   - Available compiler (gcc or clang)
#   - Platform (Linux, macOS, FreeBSD)
#   - Available libraries (hwloc, numa, hugetlbfs)

# =============================================================================
# Platform and Compiler Detection
# =============================================================================

# Detect OS
UNAME_S := $(shell uname -s)

# Auto-detect compiler: prefer gcc, fall back to clang, then cc
CC ?= $(shell command -v gcc 2>/dev/null || command -v clang 2>/dev/null || echo cc)

# Base flags (portable across gcc/clang)
CFLAGS_BASE = -O3 -Wall -Wextra -std=c11

# Platform-specific adjustments
# Note: Platform is auto-detected in code, these are just for build optimization
ifeq ($(UNAME_S),Darwin)
    # macOS: packages typically in /opt/homebrew (ARM) or /usr/local (Intel)
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),arm64)
        CFLAGS_ARCH = -mcpu=native
        CFLAGS_PATHS = -I/opt/homebrew/include
        LDFLAGS_PATHS = -L/opt/homebrew/lib
    else
        CFLAGS_ARCH = -march=native
        CFLAGS_PATHS = -I/usr/local/include
        LDFLAGS_PATHS = -L/usr/local/lib
    endif
    LDFLAGS_BASE = -pthread -lm
else ifeq ($(UNAME_S),FreeBSD)
    # FreeBSD: packages in /usr/local
    CFLAGS_ARCH = -march=native
    CFLAGS_PATHS = -I/usr/local/include
    LDFLAGS_PATHS = -L/usr/local/lib
    LDFLAGS_BASE = -pthread -lm
else ifeq ($(UNAME_S),OpenBSD)
    CFLAGS_ARCH = -march=native
    CFLAGS_PATHS = -I/usr/local/include
    LDFLAGS_PATHS = -L/usr/local/lib
    LDFLAGS_BASE = -pthread -lm
else ifeq ($(UNAME_S),NetBSD)
    CFLAGS_ARCH = -march=native
    CFLAGS_PATHS = -I/usr/local/include -I/usr/pkg/include
    LDFLAGS_PATHS = -L/usr/local/lib -L/usr/pkg/lib
    LDFLAGS_BASE = -pthread -lm
else
    # Linux (default)
    CFLAGS_ARCH = -march=native
    CFLAGS_PATHS =
    LDFLAGS_PATHS =
    LDFLAGS_BASE = -pthread -lm
endif

CFLAGS = $(CFLAGS_BASE) $(CFLAGS_ARCH) $(CFLAGS_PATHS)
LDFLAGS = $(LDFLAGS_BASE) $(LDFLAGS_PATHS)

# =============================================================================
# Library Detection
# =============================================================================

# Check for hwloc (cross-platform)
HAVE_HWLOC := $(shell pkg-config --exists hwloc 2>/dev/null && echo yes || \
                      (test -f /usr/include/hwloc.h && echo yes) || \
                      (test -f /usr/local/include/hwloc.h && echo yes) || \
                      (test -f /opt/homebrew/include/hwloc.h && echo yes))

# Check for libnuma (Linux only)
ifeq ($(UNAME_S),Linux)
    HAVE_NUMA := $(shell pkg-config --exists numa 2>/dev/null && echo yes || \
                         test -f /usr/include/numa.h && echo yes)
else
    HAVE_NUMA := no
endif

# Check for libhugetlbfs (Linux only)
ifeq ($(UNAME_S),Linux)
    HAVE_HUGETLBFS := $(shell pkg-config --exists hugetlbfs 2>/dev/null && echo yes || \
                              test -f /usr/include/hugetlbfs.h && echo yes)
else
    HAVE_HUGETLBFS := no
endif

# =============================================================================
# Build Targets
# =============================================================================

SRC = membench.c
TARGET = membench
TARGET_BASIC = membench-basic
TARGET_HWLOC = membench-hwloc
TARGET_NUMA = membench-numa
TARGET_FULL = membench-full

.PHONY: default all clean test basic hwloc numa full help info

# Default: auto-detect and build optimal version
default: $(TARGET)

# Auto-detect features and compile with all available
DETECTED_DEFS =
DETECTED_LIBS =

ifeq ($(HAVE_HUGETLBFS),yes)
    DETECTED_DEFS += -DHAVE_HUGETLBFS
    DETECTED_LIBS += -lhugetlbfs
endif

ifeq ($(HAVE_HWLOC),yes)
    DETECTED_DEFS += -DUSE_HWLOC
    DETECTED_LIBS += -lhwloc
endif

ifeq ($(HAVE_NUMA),yes)
    DETECTED_DEFS += -DUSE_NUMA
    DETECTED_LIBS += -lnuma
endif

$(TARGET): $(SRC)
	@echo "Building for $(UNAME_S) with auto-detected features..."
	@echo "  Compiler: $(CC)"
	@echo "  hwloc: $(HAVE_HWLOC)"
	@echo "  numa: $(HAVE_NUMA)"
	@echo "  hugetlbfs: $(HAVE_HUGETLBFS)"
	$(CC) $(CFLAGS) $(DETECTED_DEFS) -o $@ $< $(LDFLAGS) $(DETECTED_LIBS)

# Basic: minimal build, no optional dependencies
basic: $(TARGET_BASIC)

$(TARGET_BASIC): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Build with hwloc support (portable cache/topology detection)
hwloc: $(TARGET_HWLOC)

$(TARGET_HWLOC): $(SRC)
ifeq ($(HAVE_HWLOC),yes)
	$(CC) $(CFLAGS) -DUSE_HWLOC -o $@ $< $(LDFLAGS) -lhwloc
else
	@echo "Error: hwloc 2.x not found. Install with:"
	@echo "  Linux:  apt install libhwloc-dev  (or: yum install hwloc-devel)"
	@echo "  macOS:  brew install hwloc"
	@echo "  BSD:    pkg install hwloc2"
	@exit 1
endif

# Build with NUMA support (Linux only)
numa: $(TARGET_NUMA)

$(TARGET_NUMA): $(SRC)
ifeq ($(UNAME_S),Linux)
ifeq ($(HAVE_NUMA),yes)
	$(CC) $(CFLAGS) -DUSE_NUMA -o $@ $< $(LDFLAGS) -lnuma
else
	@echo "Error: libnuma not found. Install with:"
	@echo "  apt install libnuma-dev  (or: yum install numactl-devel)"
	@exit 1
endif
else
	@echo "Error: NUMA support is only available on Linux"
	@exit 1
endif

# Build with all features (recommended for production Linux servers)
full: $(TARGET_FULL)

$(TARGET_FULL): $(SRC)
ifeq ($(UNAME_S),Linux)
	$(CC) $(CFLAGS) -DUSE_HWLOC -DUSE_NUMA $(if $(filter yes,$(HAVE_HUGETLBFS)),-DHAVE_HUGETLBFS) \
		-o $@ $< $(LDFLAGS) -lhwloc -lnuma $(if $(filter yes,$(HAVE_HUGETLBFS)),-lhugetlbfs)
else
	@echo "Note: Building without NUMA (not available on $(UNAME_S))"
	$(CC) $(CFLAGS) -DUSE_HWLOC -o $@ $< $(LDFLAGS) -lhwloc
endif

# Build all versions that can be built on this platform
all: $(TARGET) $(TARGET_BASIC)
ifeq ($(HAVE_HWLOC),yes)
	$(MAKE) hwloc
endif
ifeq ($(HAVE_NUMA),yes)
	$(MAKE) numa
endif
ifeq ($(UNAME_S),Linux)
ifeq ($(HAVE_HWLOC),yes)
ifeq ($(HAVE_NUMA),yes)
	$(MAKE) full
endif
endif
endif

# Quick test (30 seconds)
test: $(TARGET)
	./$(TARGET) -v -t 30

# Test with hwloc
test-hwloc: $(TARGET_HWLOC)
	./$(TARGET_HWLOC) -v -t 30

# Test with NUMA
test-numa: $(TARGET_NUMA)
	./$(TARGET_NUMA) -v -t 30

# Test with full features
test-full: $(TARGET_FULL)
	./$(TARGET_FULL) -v -t 30

clean:
	rm -f $(TARGET) $(TARGET_BASIC) $(TARGET_HWLOC) $(TARGET_NUMA) $(TARGET_FULL)

# Install to /usr/local/bin
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/membench

# Show detected configuration
info:
	@echo "Platform Detection:"
	@echo "  OS:        $(UNAME_S)"
	@echo "  Compiler:  $(CC)"
	@echo "  CFLAGS:    $(CFLAGS)"
	@echo "  LDFLAGS:   $(LDFLAGS)"
	@echo ""
	@echo "Library Detection:"
	@echo "  hwloc:     $(HAVE_HWLOC)"
	@echo "  numa:      $(HAVE_NUMA)"
	@echo "  hugetlbfs: $(HAVE_HUGETLBFS)"

# Help target
help:
	@echo "sc-membench - Portable Memory Benchmark"
	@echo ""
	@echo "Build targets:"
	@echo "  make          - Auto-detect features and build optimal version"
	@echo "  make basic    - Minimal build (no optional dependencies)"
	@echo "  make hwloc    - With hwloc (portable cache detection)"
	@echo "  make numa     - With NUMA support (Linux only)"
	@echo "  make full     - With all features (hwloc + numa, Linux recommended)"
	@echo "  make all      - Build all available versions"
	@echo "  make info     - Show detected platform and libraries"
	@echo ""
	@echo "Optional dependencies:"
	@echo "  hwloc 2:   Portable cache/topology detection (requires hwloc 2.x)"
	@echo "             Linux: apt install libhwloc-dev"
	@echo "             macOS: brew install hwloc"
	@echo "             BSD:   pkg install hwloc2"
	@echo ""
	@echo "  numa:      NUMA-aware memory allocation (Linux only)"
	@echo "             apt install libnuma-dev"
	@echo ""
	@echo "  hugetlbfs: Better huge page detection (Linux only)"
	@echo "             apt install libhugetlbfs-dev"
