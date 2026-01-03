# sc-membench - Memory Bandwidth Benchmark

A portable, multi-platform memory bandwidth benchmark designed for comprehensive system analysis.

## Features

- **Multi-platform**: Works on x86, arm64, and other architectures
- **Multiple operations**: Measures read, write, copy bandwidth + memory latency
- **NUMA-aware**: Automatically handles NUMA systems (optional, works on non-NUMA too)
- **Cache analysis**: Sweeps through L1, L2, L3 cache sizes and main memory
- **Total memory model**: Reports actual total memory footprint, not per-thread size
- **Thread optimization**: Finds optimal thread/buffer configuration for each total size
- **Latency measurement**: True memory latency using pointer chasing
- **CSV output**: Machine-readable output for analysis

## Quick Start

```bash
# Compile
make

# Run with default settings (10 minute max runtime)
./membench

# Run with verbose output and custom runtime
./membench -v -t 300

# Compile with NUMA support (requires libnuma-dev)
make numa
./membench-numa -v
```

## Build Options

```bash
make              # Basic version (sysfs cache detection, Linux only)
make hwloc        # With hwloc (recommended - portable cache detection)
make numa         # With NUMA support
make full         # With hwloc + NUMA (recommended for servers)
make all          # Build all versions
make clean        # Remove built files
make test         # Quick 30-second test run
```

### Recommended Build

For production use on servers, build with all features:

```bash
# Install dependencies first
sudo apt-get install libhwloc-dev libnuma-dev  # Debian/Ubuntu
# or: sudo yum install hwloc-devel numactl-devel  # RHEL/CentOS

# Build with full features
make full
./membench-full -v
```

## Usage

```
sc-membench - Memory Bandwidth Benchmark

Usage: ./membench [options]

Options:
  -h          Show help
  -v          Verbose output (to stderr)
  -s SIZE_KB  Test only this total size (in KB), e.g. -s 1024 for 1MB
  -f          Full sweep (test all sizes up to 50% RAM)
              Default: test up to 512 MB
  -t SECONDS  Maximum runtime, 0 = unlimited (default: unlimited)
```

## Output Format

CSV output to stdout with columns:

| Column | Description |
|--------|-------------|
| `size_kb` | **Total** memory footprint tested (KB) |
| `operation` | Operation type: `read`, `write`, `copy`, or `latency` |
| `bandwidth_mb_s` | Bandwidth achieved (MB/s), 0 for latency test |
| `latency_ns` | Memory latency (nanoseconds), 0 for bandwidth tests |
| `threads` | Thread count that achieved best result |
| `iterations` | Number of iterations performed |
| `elapsed_s` | Elapsed time for the test (seconds) |

### Example Output

```csv
size_kb,operation,bandwidth_mb_s,latency_ns,threads,iterations,elapsed_s
4,read,8000000.00,0,16,10000000,0.08
4,write,4000000.00,0,16,6000000,0.10
16384,read,2200000.00,0,16,100000,0.12
16384,write,3000000.00,0,32,150000,0.16
1048576,read,115000.00,0,32,275,2.45
1048576,write,68000.00,0,32,188,2.83
```

## Operations Explained

### Read (`read`)
Reads all 64-bit words from the buffer using XOR (faster than addition, no carry chains). This measures pure read bandwidth.

```c
checksum ^= buffer[i];  // For all elements, using 8 independent accumulators
```

### Write (`write`)
Writes a pattern to all 64-bit words in the buffer. This measures pure write bandwidth.

```c
buffer[i] = pattern;  // For all elements
```

### Copy (`copy`)
Copies data from source to destination buffer. Reports bandwidth as `buffer_size / time` (matching lmbench's approach), not `2 × buffer_size / time`.

```c
dst[i] = src[i];  // For all elements
```

**Note:** Copy bandwidth is typically lower than read or write alone because it performs both operations. The reported bandwidth represents the buffer size traversed, not total bytes moved (read + write).

### Latency (`latency`)
Measures true memory access latency using **pointer chasing**. Each memory access depends on the previous one, preventing CPU pipelining and prefetching.

```c
// Each load depends on previous (can't be optimized away)
p = *p;  // Address comes from previous load
p = *p;
p = *p;
// ...
```

The buffer is initialized as a linked list of pointers in **randomized order** to defeat hardware prefetchers. This measures:
- L1/L2/L3 cache hit latency at small sizes
- DRAM access latency at large sizes
- True memory latency without pipelining effects

Results are reported in **nanoseconds per access**, not MB/s.

**Large L3 cache support**: The latency test uses buffers up to 2GB (or 25% of RAM) to correctly measure DRAM latency even on processors with huge L3 caches like AMD EPYC 9754 (1.1GB L3 with 3D V-Cache).

**Adaptive early termination**: For buffer sizes >1.5GB in series mode, when DRAM latency is detected (>50ns), the benchmark uses adaptive termination to stop early once measurements stabilize. This helps on systems with both large caches and slower DRAM.

## Memory Sizes Tested

The benchmark tests **total memory** sizes starting from an **adaptive minimum** based on detected cache topology:

- **Minimum size**: L1_cache × num_CPUs (e.g., 48KB × 192 CPUs = 9MB on r8a.48xlarge)
- **Maximum size**: 512MB (default) or up to 50% of RAM (with `-f` flag)

### Why Adaptive Minimum?

To measure aggregate system bandwidth (not per-core), we need enough total memory so each thread can have a meaningful buffer. The formula `L1 × num_CPUs` ensures:

- On 16-core system (48KB L1): minimum = 768KB → can use 16 threads × 48KB each
- On 192-core system (48KB L1): minimum = 9MB → can use 192 threads × 48KB each

This ensures bandwidth measurements scale properly with core count.

### Cache Detection

With hwloc (recommended), cache sizes are detected automatically on any platform.
Without hwloc, the benchmark parses `/sys/devices/system/cpu/*/cache/` (Linux only).

## Thread Scaling and Total Memory Model

For each **total memory size**, the benchmark tries different thread/buffer configurations while keeping the total memory footprint constant:

```
Example for 1MB total (read/write):
  1 thread  × 1MB buffer   = 1MB total
  2 threads × 512KB buffer = 1MB total  
  4 threads × 256KB buffer = 1MB total

Example for 1MB total (copy - needs src + dst):
  1 thread  × 512KB src + 512KB dst = 1MB total
  2 threads × 256KB src + 256KB dst = 1MB total
```

The benchmark finds the optimal configuration and reports:
- `size_kb`: The total memory footprint (not per-thread)
- `threads`: The winning thread count
- `bandwidth_mb_s`: The best bandwidth achieved

This approach ensures that **size_kb values are directly comparable** - a 1MB test and a 16GB test both represent actual total memory usage, making the results meaningful for comparing cache vs main memory performance.

### Why This Matters

With per-thread buffers, more threads means more total memory. If we reported per-thread size:
- "1MB" with 24 threads = 24MB actual memory
- "16GB" with 1 thread = 16GB actual memory

These wouldn't be comparable! By keeping total memory constant and varying thread/buffer splits, we find the true optimal configuration for each memory footprint.

### What the Benchmark Discovers

- **Small sizes (cache-resident)**: Many threads with small buffers often wins
- **Large sizes (main memory)**: Fewer threads with larger buffers may be optimal
- **NUMA systems**: Thread distribution across nodes affects performance

## NUMA Support

When compiled with `-DUSE_NUMA` and linked with `-lnuma`:

- Detects NUMA topology automatically
- Distributes threads across NUMA nodes
- Interleaves memory allocation for fair testing
- Works transparently on single-node systems

## Comparison with lmbench

### Bandwidth (bw_mem)

| Aspect | sc-membench | lmbench bw_mem |
|--------|-------------|----------------|
| **Parallelism model** | Threads | Processes (fork) |
| **Buffer allocation** | Each thread has own buffer | Each process has own buffer |
| **Size reporting** | Total memory footprint | Per-process buffer size |
| **Read operation** | Reads 100% of data | `rd` reads 25% (strided) |
| **Copy reporting** | Buffer size / time | Buffer size / time |

**Key differences:**

1. **Size meaning**: sc-membench's `size_kb` is total memory used; bw_mem's size is per-process
2. **Read operation**: bw_mem `rd` uses strided access (reads 25% of data), reporting ~4x higher apparent bandwidth
3. **Thread optimization**: sc-membench finds optimal thread/buffer configuration for each total size

### Latency (lat_mem_rd)

sc-membench's `latency` operation is comparable to lmbench's `lat_mem_rd`:

| Aspect | sc-membench latency | lmbench lat_mem_rd |
|--------|---------------------|-------------------|
| **Method** | Pointer chasing | Pointer chasing |
| **Order** | Randomized | Configurable stride |
| **Output** | Nanoseconds | Nanoseconds |

Both measure true memory latency by using dependent loads that prevent pipelining and prefetching.

## Interpreting Results

### Cache Effects
Look for bandwidth drops and latency increases as sizes exceed cache levels:
- Dramatic change at L1 boundary (32-64KB typically)
- Another change at L2 boundary (256KB-1MB typically)
- Final change at L3 boundary (8-64MB typically)

### Thread Configuration
- Small total sizes: More threads with small buffers (fits in per-core cache)
- Large total sizes: Fewer threads with larger buffers (memory-bound)
- NUMA systems: Thread distribution across nodes affects performance
- Latency test: Always uses 1 thread with full buffer size

### Bandwidth Values
Typical modern systems:
- L1 cache: 200-500 GB/s (varies with frequency)
- L2 cache: 100-200 GB/s
- L3 cache: 50-100 GB/s
- Main memory: 20-100 GB/s (DDR4/DDR5, depends on channels)

### Latency Values
Typical modern systems:
- L1 cache: 1-2 ns
- L2 cache: 3-10 ns
- L3 cache: 10-40 ns (larger/3D V-Cache may be higher)
- Main memory: 25-50 ns (fast DDR5) to 60-120 ns (DDR4)

## Dependencies

- **Required**: POSIX threads (pthread), C11 compiler
- **Recommended**: libhwloc for portable cache topology detection
- **Optional**: libnuma for NUMA support

### Installing Dependencies

```bash
# Debian/Ubuntu
apt-get install build-essential libhwloc-dev libnuma-dev

# RHEL/CentOS/Fedora
yum install gcc make hwloc-devel numactl-devel

# macOS (hwloc only, no NUMA)
brew install hwloc
xcode-select --install
```

### What Each Dependency Provides

| Library | Purpose | Platforms |
|---------|---------|-----------|
| **hwloc** | Cache topology detection (L1/L2/L3 sizes) | Linux, macOS, Windows, BSD |
| **libnuma** | NUMA-aware memory allocation | Linux only |

Without hwloc, the benchmark falls back to parsing `/sys/devices/system/cpu/*/cache/` (Linux only).
Without libnuma, memory is allocated without NUMA awareness (may underperform on multi-socket systems).

## License

Apache 2.0

## See Also

- [STREAM benchmark](https://www.cs.virginia.edu/stream/)
- [lmbench](https://sourceforge.net/projects/lmbench/)

