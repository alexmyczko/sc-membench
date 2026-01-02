# sc-membench Makefile
#
# Build options:
#   make           - Build basic version (sysfs cache detection)
#   make hwloc     - Build with hwloc (recommended, portable cache detection)
#   make numa      - Build with NUMA support
#   make full      - Build with both hwloc and NUMA (recommended for servers)
#   make all       - Build all versions
#   make clean     - Remove built files
#   make test      - Quick test run
#
# Dependencies:
#   Basic:    gcc, pthread, math library (standard)
#   hwloc:    libhwloc-dev (Debian/Ubuntu), hwloc-devel (RHEL/CentOS), hwloc (macOS)
#   NUMA:     libnuma-dev (Debian/Ubuntu), numactl-devel (RHEL/CentOS)

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -march=native
LDFLAGS = -pthread -lm

# Source files
SRC = membench.c
TARGET = membench
TARGET_HWLOC = membench-hwloc
TARGET_NUMA = membench-numa
TARGET_FULL = membench-full

.PHONY: all clean test hwloc numa full

# Default: build basic version (sysfs-based cache detection)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Build with hwloc support (portable cache/topology detection)
hwloc: $(TARGET_HWLOC)

$(TARGET_HWLOC): $(SRC)
	$(CC) $(CFLAGS) -DUSE_HWLOC -o $@ $< $(LDFLAGS) -lhwloc

# Build with NUMA support
numa: $(TARGET_NUMA)

$(TARGET_NUMA): $(SRC)
	$(CC) $(CFLAGS) -DUSE_NUMA -o $@ $< $(LDFLAGS) -lnuma

# Build with both hwloc and NUMA (recommended for production)
full: $(TARGET_FULL)

$(TARGET_FULL): $(SRC)
	$(CC) $(CFLAGS) -DUSE_HWLOC -DUSE_NUMA -o $@ $< $(LDFLAGS) -lhwloc -lnuma

# Build all versions
all: $(TARGET) $(TARGET_HWLOC) $(TARGET_NUMA) $(TARGET_FULL)

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
	rm -f $(TARGET) $(TARGET_HWLOC) $(TARGET_NUMA) $(TARGET_FULL)

# Install to /usr/local/bin
install: $(TARGET_FULL)
	install -m 755 $(TARGET_FULL) /usr/local/bin/membench

# Help target
help:
	@echo "sc-membench build targets:"
	@echo "  make          - Basic version (sysfs cache detection, Linux only)"
	@echo "  make hwloc    - With hwloc (portable cache detection)"
	@echo "  make numa     - With NUMA support"
	@echo "  make full     - With hwloc + NUMA (recommended for servers)"
	@echo "  make all      - Build all versions"
	@echo ""
	@echo "Dependencies:"
	@echo "  hwloc: apt install libhwloc-dev  (or: yum install hwloc-devel)"
	@echo "  NUMA:  apt install libnuma-dev   (or: yum install numactl-devel)"
