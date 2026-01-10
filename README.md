# sc-membench - Memory Bandwidth Benchmark

A portable, multi-platform memory bandwidth benchmark designed for comprehensive system analysis.

## Features

- **Multi-platform**: Works on x86, arm64, and other architectures
- **Multiple operations**: Measures read, write, copy bandwidth + memory latency
- **NUMA-aware**: Automatically handles NUMA systems (optional, works on non-NUMA too)
- **Cache-aware sizing**: Adaptive test sizes based on detected L1, L2, L3 cache hierarchy
- **Per-thread buffer model**: Like bw_mem, each thread gets its own buffer
- **Thread control**: Default uses all CPUs; optional auto-scaling to find optimal thread count
- **Latency measurement**: True memory latency using pointer chasing
- **Best-of-N runs**: Each test runs multiple times, reports best result (like lmbench)
- **CSV output**: Machine-readable output for analysis

## Quick Start

```bash
# Compile
make

# Run with default settings (uses all CPUs, cache-aware sizes)
./membench

# Run with verbose output and 5 minute time limit
./membench -v -t 300

# Test specific buffer size (1MB per thread)
./membench -s 1024

# Compile with NUMA support (requires libnuma-dev)
make numa
./membench-numa -v
```

## Docker Usage

The easiest way to run sc-membench without building is using the pre-built Docker image:

```bash
# Run with default settings
docker run --rm ghcr.io/sparecores/membench:main

# Run with verbose output and time limit
docker run --rm ghcr.io/sparecores/membench:main -v -t 300

# Test specific buffer size
docker run --rm ghcr.io/sparecores/membench:main -s 1024

# Recommended: use --privileged and huge pages for best accuracy
docker run --rm --privileged ghcr.io/sparecores/membench:main -H -v

# Save output to file
docker run --rm --privileged ghcr.io/sparecores/membench:main -H > results.csv
```

**Notes:**
- The `--privileged` flag is recommended for optimal CPU pinning and NUMA support
- The `-H` flag enables huge pages automatically for large buffers (≥ 4MB), no setup required

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
  -v          Verbose output (use -vv for more detail)
  -s SIZE_KB  Test only this buffer size (in KB), e.g. -s 1024 for 1MB
  -r TRIES    Repeat each test N times, report best (default: 3)
  -f          Full sweep (test larger sizes up to memory limit)
  -p THREADS  Use exactly this many threads (default: num_cpus)
  -a          Auto-scaling: try different thread counts to find best
              (slower but finds optimal thread count per buffer size)
  -t SECONDS  Maximum runtime, 0 = unlimited (default: unlimited)
  -o OP       Run only this operation: read, write, copy, or latency
              Can be specified multiple times (default: all)
  -H          Enable huge pages for large buffers (>= 4MB)
              Uses THP automatically, no setup required
```

## Output Format

CSV output to stdout with columns:

| Column | Description |
|--------|-------------|
| `size_kb` | **Per-thread** buffer size (KB) |
| `operation` | Operation type: `read`, `write`, `copy`, or `latency` |
| `bandwidth_mb_s` | Aggregate bandwidth across all threads (MB/s), 0 for latency |
| `latency_ns` | Memory latency (nanoseconds), 0 for bandwidth tests |
| `threads` | Thread count used |
| `iterations` | Number of iterations performed |
| `elapsed_s` | Elapsed time for the test (seconds) |

**Total memory used** = `size_kb × threads` (or `× 2` for copy which needs src + dst).

### Example Output

```csv
size_kb,operation,bandwidth_mb_s,latency_ns,threads,iterations,elapsed_s
16,read,2276940.70,0,48,589309,0.19
16,write,1302796.66,0,48,350557,0.20
16,latency,0,1.07,1,48176,0.11
512,read,807264.98,0,48,11728,0.35
512,write,726262.08,0,48,8512,0.29
512,latency,0,4.78,1,316,0.10
65536,read,113577.83,0,48,23,0.62
65536,write,46728.81,0,48,20,1.31
65536,latency,0,65.12,1,3,1.64
```

In this example (48-core system with 32KB L1, 1MB L2, 36MB L3):
- **16KB**: Fits in L1 → very high bandwidth (~2.3 TB/s read), low latency (~1ns)
- **512KB**: Fits in L2 → good bandwidth (~800 GB/s read), moderate latency (~5ns)
- **64MB**: Past L3 → RAM bandwidth (~114 GB/s read), high latency (~65ns)

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

The benchmark tests **per-thread buffer sizes** at cache transition points, automatically adapting to the detected cache hierarchy:

### Adaptive Cache-Aware Sizes

Based on detected L1, L2, L3 cache sizes:

| Size | Purpose |
|------|---------|
| L1/2 | Pure L1 cache performance |
| L1 | L1 cache boundary |
| 2×L1 | L1→L2 transition (if fits before L2/2) |
| L2/2 | Pure L2 cache performance |
| L2 | L2 cache boundary |
| 2×L2 | L2→L3 transition (if fits before L3/2) |
| L3/2 | Mid L3 cache |
| L3 | L3 cache boundary |
| 2×L3 | Past L3, hitting RAM |
| 4×L3 | Deep into RAM |

With `-f` (full sweep), additional larger sizes are tested up to the memory limit.

### Cache Detection

With hwloc (recommended), cache sizes are detected automatically on any platform.
Without hwloc, the benchmark parses `/sys/devices/system/cpu/*/cache/` (Linux only).

If cache detection fails, sensible defaults are used (32KB L1, 256KB L2, 8MB L3).

## Thread Model (Per-Thread Buffers)

Like bw_mem, each thread gets its **own private buffer**:

```
Example for 1MB buffer size with 4 threads (read/write):
  Thread 0: 1MB buffer
  Thread 1: 1MB buffer
  Thread 2: 1MB buffer
  Thread 3: 1MB buffer
  Total memory: 4MB

Example for 1MB buffer size with 4 threads (copy):
  Thread 0: 1MB src + 1MB dst = 2MB
  Thread 1: 1MB src + 1MB dst = 2MB
  ...
  Total memory: 8MB
```

### Thread Modes

| Mode | Flag | Behavior |
|------|------|----------|
| **Default** | (none) | Use `num_cpus` threads |
| **Explicit** | `-p N` | Use exactly N threads |
| **Auto-scaling** | `-a` | Try 1, 2, 4, ..., num_cpus threads, report best |

### What the Benchmark Measures

- **Aggregate bandwidth**: Sum of all threads' bandwidth
- **Per-thread buffer**: Each thread works on its own memory region
- **No sharing**: Threads don't contend for the same cache lines

### Interpreting Results

- `size_kb` = buffer size per thread
- `threads` = number of threads used
- `bandwidth_mb_s` = total system bandwidth (all threads combined)
- Total memory = `size_kb × threads` (×2 for copy)

## NUMA Support

When compiled with `-DUSE_NUMA` and linked with `-lnuma`:

- Detects NUMA topology automatically
- Maps CPUs to their NUMA nodes
- Load-balances threads across NUMA nodes
- Binds each thread's memory to its local node
- Works transparently on UMA (single-node) systems

### NUMA Load Balancing

On multi-socket systems, threads are distributed **round-robin across NUMA nodes** to ensure balanced utilization of all memory controllers.

**Example: 128 threads on a 2-node system (96 CPUs per node):**

```
Without balancing (sequential):          With balancing (round-robin):
  Thread 0-95  → Node 0 (96 threads)       Thread 0  → Node 0, CPU 0
  Thread 96-127 → Node 1 (32 threads)      Thread 1  → Node 1, CPU 96
  Result: Node 0 overloaded!               Thread 2  → Node 0, CPU 1
                                           Thread 3  → Node 1, CPU 97
                                           ...
                                           Result: 64 threads per node (balanced!)
```

**Impact:**
- ~15% higher bandwidth with balanced distribution
- More accurate measurement of total system memory bandwidth
- Exercises all memory controllers evenly

### NUMA-Local Memory

Each thread's buffer is bound to its local NUMA node using `mbind(MPOL_BIND)`:

```c
int node = numa_node_of_cpu(cpu_id);
mbind(buffer, size, MPOL_BIND, &nodemask, ...);
```

This ensures:
- Memory is allocated on the same node as the accessing CPU
- No cross-node memory access penalties
- No memory migrations during the benchmark

### Verbose Output

Use `-v` to see the detected NUMA topology:

```
NUMA: 2 nodes detected (libnuma enabled)
NUMA topology:
  Node 0: 96 CPUs (first: 0, last: 95)
  Node 1: 96 CPUs (first: 96, last: 191)
```

## Huge Pages Support

Use `-H` to enable huge pages (2MB instead of 4KB). This reduces TLB (Translation Lookaside Buffer) pressure, which is especially beneficial for:

- **Large buffer tests**: A 2GB buffer needs 512K page table entries with 4KB pages, but only 1024 with 2MB huge pages
- **Latency tests**: Random pointer-chasing access patterns cause many TLB misses with small pages
- **Accurate measurements**: TLB overhead can distort results, making memory appear slower than it is

### Automatic and smart

The `-H` option is designed to "just work":

1. **Automatic threshold**: Huge pages are only used for buffers ≥ 4MB. Smaller buffers use regular pages automatically (no wasted memory, no user intervention needed).

2. **No setup required**: The benchmark uses **Transparent Huge Pages (THP)** via `madvise(MADV_HUGEPAGE)`, which is handled automatically by the Linux kernel. No root access or pre-allocation needed.

3. **Graceful fallback**: If THP isn't available, the benchmark falls back to regular pages transparently.

### How it works

When `-H` is enabled and buffer size ≥ 4MB:

1. **First tries explicit huge pages** (`MAP_HUGETLB`) for deterministic 2MB pages
2. **Falls back to THP** (`madvise(MADV_HUGEPAGE)`) which works without pre-configuration
3. **Falls back to regular pages** if neither is available

### Optional: Pre-allocating explicit huge pages

For the most deterministic results, you can pre-allocate explicit huge pages:

```bash
# Check current huge page status
grep Huge /proc/meminfo

# Calculate huge pages needed for BANDWIDTH tests (read/write/copy):
#   threads × buffer_size × 2 (for copy: src+dst) / 2MB
#
# Examples:
#   8 CPUs,  256 MiB buffer:   8 × 256 × 2 / 2 =  2,048 pages (4 GB)
#   64 CPUs, 256 MiB buffer:  64 × 256 × 2 / 2 = 16,384 pages (32 GB)
#  192 CPUs, 256 MiB buffer: 192 × 256 × 2 / 2 = 49,152 pages (96 GB)
#
# LATENCY tests run single-threaded, so need much less:
#   256 MiB buffer: 256 / 2 = 128 pages (256 MB)

# Allocate huge pages (requires root) - adjust for your system
echo 49152 | sudo tee /proc/sys/vm/nr_hugepages

# Run with huge pages (will use explicit huge pages if available)
./membench -H -v
```

However, this is **optional** - THP works well for most use cases without any setup, and doesn't require pre-allocation. If explicit huge pages run out, the benchmark automatically falls back to THP.

### Usage recommendation

Just add `-H` to your command line - the benchmark handles everything automatically:

```bash
# Recommended for production benchmarking
./membench -H

# With verbose output to see what's happening
./membench -H -v
```

The benchmark will use huge pages only where they help (large buffers) and regular pages where they don't (small buffers).

### Why latency improves more than bandwidth

You may notice that `-H` dramatically improves latency measurements (often 20-40% lower) while bandwidth stays roughly the same. This is expected:

**Latency tests** use pointer chasing - random jumps through memory. Each access requires address translation via the TLB (Translation Lookaside Buffer):

| Buffer Size | 4KB pages | 2MB huge pages |
|-------------|-----------|----------------|
| 128 MB | 32,768 pages | 64 pages |
| TLB fit? | No (TLB ~1000-2000 entries) | Yes |
| TLB misses | Frequent | Rare |

With 4KB pages on a 128MB buffer:
- 32,768 pages can't fit in the TLB
- Random pointer chasing causes frequent TLB misses
- Each TLB miss adds **10-20+ CPU cycles** (page table walk)
- Measured latency = true memory latency + TLB overhead

With 2MB huge pages:
- Only 64 pages easily fit in the TLB
- Almost no TLB misses
- Measured latency ≈ **true memory latency**

Example results on a typical system:
```
# Without huge pages (includes TLB overhead)
131072,latency,0,91.11,1,3,4.59

# With huge pages (true memory latency)
131072,latency,0,63.27,1,3,3.18
```

The ~63 ns is the more accurate measurement - the ~91 ns included ~28 ns of TLB miss overhead.

**Bandwidth tests** don't improve as much because:
- Sequential access has better TLB locality (same pages accessed repeatedly)
- Hardware prefetchers hide TLB miss latency
- The memory bus is already saturated

**Bottom line**: Use `-H` for accurate latency measurements. The latency without `-H` includes TLB overhead that isn't really "memory latency."

## Consistent Results

Achieving consistent benchmark results on modern multi-core systems requires careful handling of:

### Thread Pinning

Each thread is pinned to a specific CPU using `pthread_setaffinity_np()`. This prevents the OS scheduler from migrating threads between cores, which causes huge variability.

### NUMA-Aware Memory

On NUMA systems, memory is bound to the local NUMA node of each thread's CPU using `mbind(MPOL_BIND)`. This ensures:
- Memory is close to where it will be accessed
- No cross-node memory access penalties
- No memory migrations during the benchmark

### Best-of-N Runs

Like lmbench (TRIES=11), each test configuration runs multiple times and reports the best result:

1. First run is a warmup (discarded) to stabilize CPU frequency
2. Each configuration is then tested 3 times (configurable with `-r`)
3. For bandwidth: highest bandwidth is reported
4. For latency: lowest latency is reported

### Result

With these optimizations, benchmark variability is typically **<1%** (compared to 30-60% without them).

### Configuration

```bash
./membench -r 5    # Run each test 5 times instead of 3
./membench -r 1    # Single run (fastest, still consistent due to pinning)
./membench -p 16   # Use exactly 16 threads
./membench -a      # Auto-scale to find optimal thread count
```

## Comparison with lmbench

### Bandwidth (bw_mem)

| Aspect | sc-membench | lmbench bw_mem |
|--------|-------------|----------------|
| **Parallelism model** | Threads | Processes (fork) |
| **Buffer allocation** | Each thread has own buffer | Each process has own buffer |
| **Size reporting** | Per-thread buffer size | Per-process buffer size |
| **Read operation** | Reads 100% of data | `rd` reads 25% (strided) |
| **Copy reporting** | Buffer size / time | Buffer size / time |

**Key differences:**

1. **Size meaning**: Both report per-worker buffer size (comparable)
2. **Read operation**: bw_mem `rd` uses strided access (reads 25% of data), reporting ~4x higher apparent bandwidth. sc-membench reads 100% of data.
3. **Thread control**: sc-membench defaults to num_cpus threads; use `-a` for auto-scaling or `-p N` for explicit count

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
Look for bandwidth drops and latency increases as buffer sizes exceed cache levels:
- Dramatic change at L1 boundary (32-64KB per thread typically)
- Another change at L2 boundary (256KB-1MB per thread typically)
- Final change when total memory exceeds L3 (depends on thread count)

### Thread Configuration
- By default, all CPUs are used for maximum aggregate bandwidth
- Use `-p N` to test with a specific thread count
- Use `-a` to find optimal thread count (slower but thorough)
- Latency test: Always uses 1 thread (measures true access latency)

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

Mozilla Public License 2.0

## See Also

- [STREAM benchmark](https://www.cs.virginia.edu/stream/)
- [lmbench](https://sourceforge.net/projects/lmbench/)

