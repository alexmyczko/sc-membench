# sc-membench - Memory Bandwidth Benchmark

A portable, multi-platform memory bandwidth benchmark designed for comprehensive system analysis.

## Features

- **Multi-platform**: Works on x86, arm64, and other architectures
- **Multiple operations**: Measures read, write, copy bandwidth + memory latency
- **NUMA-aware**: Automatically handles NUMA systems (optional, works on non-NUMA too)
- **Cache analysis**: Sweeps through L1, L2, L3 cache sizes and main memory
- **Thread scaling**: Finds optimal thread count for peak bandwidth
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
make              # Build without NUMA support
make numa         # Build with NUMA support
make all          # Build both versions
make clean        # Remove built files
make test         # Quick 30-second test run
```

## Usage

```
sc-membench - Memory Bandwidth Benchmark

Usage: ./membench [options]

Options:
  -h          Show help
  -v          Verbose output (to stderr)
  -t SECONDS  Maximum runtime (default: 600)
```

## Output Format

CSV output to stdout with columns:

| Column | Description |
|--------|-------------|
| `size_bytes` | Memory buffer size tested (bytes) |
| `operation` | Operation type: `read`, `write`, `copy`, or `latency` |
| `bandwidth_mb_s` | Bandwidth achieved (MB/s), 0 for latency test |
| `latency_ns` | Memory latency (nanoseconds), 0 for bandwidth tests |
| `threads` | Thread count that achieved best result |
| `iterations` | Number of iterations performed |
| `elapsed_s` | Elapsed time for the test (seconds) |

### Example Output

```csv
size_bytes,operation,bandwidth_mb_s,latency_ns,threads,iterations,elapsed_s
1024,read,45678.90,0,1,1000000,0.012345
1024,write,34567.89,0,1,1000000,0.015678
1024,copy,56789.01,0,2,500000,0.018901
1024,latency,0,1.23,1,1000000,0.012345
...
1073741824,read,98765.43,0,64,10,0.109876
1073741824,write,87654.32,0,64,10,0.123456
1073741824,copy,76543.21,0,128,10,0.145678
1073741824,latency,0,85.67,1,10,0.109876
```

## Operations Explained

### Read (`read`)
Reads all 64-bit words from the buffer and sums them. This measures pure read bandwidth.

```c
sum += buffer[i];  // For all elements
```

### Write (`write`)
Writes a pattern to all 64-bit words in the buffer. This measures pure write bandwidth.

```c
buffer[i] = pattern;  // For all elements
```

### Copy (`copy`)
Copies data from source to destination buffer. This measures read+write bandwidth combined.

```c
dst[i] = src[i];  // For all elements
```

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

## Memory Sizes Tested

The benchmark automatically tests these sizes (up to 50% of available RAM):

- **L1 cache range**: 1KB - 64KB
- **L2 cache range**: 96KB - 1MB
- **L3 cache range**: 1.5MB - 256MB
- **Main memory**: 384MB - 128GB+

## Thread Scaling

For each size and operation, the benchmark:
1. Tries various thread counts (1, 2, 4, 8, 16, 32, ... up to 2×nproc)
2. Reports the thread count that achieved the best bandwidth
3. Uses early termination when bandwidth clearly drops

This helps identify:
- Optimal thread count for different memory regions
- NUMA effects (where more threads across nodes help)
- Contention (where too many threads hurt performance)

## NUMA Support

When compiled with `-DUSE_NUMA` and linked with `-lnuma`:

- Detects NUMA topology automatically
- Distributes threads across NUMA nodes
- Interleaves memory allocation for fair testing
- Works transparently on single-node systems

## Comparison with lmbench

### Bandwidth (bw_mem)

sc-membench measures **actual data bandwidth** while lmbench's `bw_mem rd` uses strided access:

| Aspect | sc-membench | lmbench bw_mem |
|--------|-------------|----------------|
| **Parallelism model** | Threads (shared memory) | Processes (fork) |
| **Buffer allocation** | Each thread has own buffer | Each process has own buffer |
| **Read operation** | Reads 100% of data | `rd` reads 25% (strided) |
| **Bandwidth metric** | Actual bytes transferred | Buffer size traversed |

For small cache-resident sizes, `bw_mem rd` reports ~4x higher bandwidth because it only reads 25% of data but reports the full buffer size.

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

### Thread Scaling
- Small sizes: 1-4 threads often optimal (cache-resident, low contention)
- Large sizes: Many threads needed to saturate memory bandwidth
- NUMA systems: Need threads on all nodes for full bandwidth
- Latency test: Always uses 1 thread (latency doesn't benefit from parallelism)

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
- L3 cache: 10-30 ns
- Main memory: 60-120 ns (DDR4/DDR5)

## Dependencies

- **Required**: POSIX threads (pthread), C11 compiler
- **Optional**: libnuma for NUMA support

### Installing Dependencies

```bash
# Debian/Ubuntu
apt-get install build-essential libnuma-dev

# RHEL/CentOS
yum install gcc make numactl-devel

# macOS (no NUMA support)
xcode-select --install
```

## License

Apache 2.0

## See Also

- [STREAM benchmark](https://www.cs.virginia.edu/stream/)
- [lmbench](https://sourceforge.net/projects/lmbench/)

