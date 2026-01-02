# sc-membench Makefile
#
# Build options:
#   make           - Build without NUMA support
#   make numa      - Build with NUMA support
#   make all       - Build both versions
#   make clean     - Remove built files
#   make test      - Quick test run

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -march=native
LDFLAGS = -pthread -lm

# Source files
SRC = membench.c
TARGET = membench
TARGET_NUMA = membench-numa

.PHONY: all clean test numa

# Default: build without NUMA
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Build with NUMA support
numa: $(TARGET_NUMA)

$(TARGET_NUMA): $(SRC)
	$(CC) $(CFLAGS) -DUSE_NUMA -o $@ $< $(LDFLAGS) -lnuma

# Build both versions
all: $(TARGET) $(TARGET_NUMA)

# Quick test (30 seconds)
test: $(TARGET)
	./$(TARGET) -v -t 30

# Test with NUMA
test-numa: $(TARGET_NUMA)
	./$(TARGET_NUMA) -v -t 30

clean:
	rm -f $(TARGET) $(TARGET_NUMA)

# Install to /usr/local/bin
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# For cross-compilation
arm64:
	$(CC) $(CFLAGS) -o $(TARGET)-arm64 $(SRC) $(LDFLAGS)

x86_64:
	$(CC) $(CFLAGS) -o $(TARGET)-x86_64 $(SRC) $(LDFLAGS)

