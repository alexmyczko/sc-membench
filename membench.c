/*
 * sc-membench - Portable Memory Bandwidth Benchmark
 *
 * A multi-platform memory bandwidth benchmark that:
 * - Works on x86, arm64, and other architectures
 * - Measures read, write, and copy bandwidth
 * - Handles NUMA automatically (works on non-NUMA too)
 * - Sweeps through cache and memory sizes
 * - Finds optimal thread count for peak bandwidth
 * - Outputs CSV format for analysis
 *
 * Compile:
 *   gcc -O3 -pthread -o membench membench.c
 *   # With NUMA support (optional):
 *   gcc -O3 -pthread -DUSE_NUMA -o membench membench.c -lnuma
 *
 * Usage:
 *   ./membench [options]
 *   ./membench -h   # Show help
 *
 * Copyright 2026 Spare Cores
 * Licensed under Apache 2.0
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <sys/mman.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define VERSION "1.0.0"

/* Target time per individual measurement (seconds) */
#define TARGET_TIME_PER_TEST 0.1

/* Minimum iterations per test */
#define MIN_ITERATIONS 3

/* Maximum iterations per test */
#define MAX_ITERATIONS 10000000

/* Default total runtime target (seconds). 0 = unlimited */
#define DEFAULT_MAX_RUNTIME 0

/* Memory sizes to test (in bytes) - key points to show cache hierarchy
 * ~20 sizes covering L1, L2, L3, and main memory */
static const size_t DEFAULT_SIZES[] = {
    /* L1 cache range (typically 32-64KB per core) */
    4096,           /* 4 KB */
    16384,          /* 16 KB */
    32768,          /* 32 KB */
    65536,          /* 64 KB */
    
    /* L2 cache range (typically 256KB-1MB per core) */
    131072,         /* 128 KB */
    262144,         /* 256 KB */
    524288,         /* 512 KB */
    1048576,        /* 1 MB */
    
    /* L3 cache range (typically 8-256MB shared) */
    2097152,        /* 2 MB */
    4194304,        /* 4 MB */
    8388608,        /* 8 MB */
    16777216,       /* 16 MB */
    33554432,       /* 32 MB */
    67108864,       /* 64 MB */
    134217728,      /* 128 MB */
    268435456,      /* 256 MB */
    
    /* Main memory */
    536870912,      /* 512 MB */
    1073741824,     /* 1 GB */
    2147483648UL,   /* 2 GB */
    4294967296UL,   /* 4 GB */
    8589934592UL,   /* 8 GB */
    17179869184UL,  /* 16 GB */
    34359738368UL,  /* 32 GB */
    68719476736UL,  /* 64 GB */
    137438953472UL, /* 128 GB */
    0  /* sentinel */
};

/* ============================================================================
 * Types
 * ============================================================================ */

typedef enum {
    OP_READ,
    OP_WRITE,
    OP_COPY,
    OP_LATENCY   /* Memory latency test using pointer chasing */
} operation_t;

static const char* OP_NAMES[] = {"read", "write", "copy", "latency"};

typedef struct {
    void *src;              /* Source buffer (thread-private) */
    void *dst;              /* Destination buffer for copy (thread-private) */
    size_t size;            /* Buffer size per thread */
    operation_t op;         /* Operation type */
    int iterations;         /* Number of iterations */
    uint64_t checksum;      /* Prevent optimization */
    double elapsed;         /* Thread's elapsed time */
    int thread_id;          /* Logical thread ID (0..nthreads-1) */
    int cpu_id;             /* Physical CPU ID to pin to */
    int numa_node;          /* NUMA node to bind to (-1 for no binding) */
    int ready;              /* Thread ready flag */
} thread_work_t;

typedef struct {
    size_t size;
    operation_t op;
    int threads;
    double bandwidth_mb_s;  /* For read/write/copy */
    double latency_ns;      /* For latency test */
    double elapsed_s;
    int iterations;
} result_t;

/* ============================================================================
 * Global state
 * ============================================================================ */

static pthread_barrier_t g_barrier;
static volatile int g_running = 1;
static int g_verbose = 0;
static int g_full_sweep = 0;      /* If 1, test all sizes up to max; if 0, stop early when converged */
static size_t g_single_size = 0;  /* If > 0, test only this size (in bytes) */
static int g_num_cpus = 0;
static int g_numa_nodes = 0;
static size_t g_total_memory = 0;

/* NUMA topology - CPUs per node for balanced thread distribution */
#define MAX_NUMA_NODES 64
#define MAX_CPUS_PER_NODE 512
static int g_cpus_per_node[MAX_NUMA_NODES];           /* Count of CPUs on each node */
static int g_node_cpus[MAX_NUMA_NODES][MAX_CPUS_PER_NODE];  /* CPU IDs for each node */
/* Number of times to run each benchmark, taking best result (like lmbench TRIES=11) */
#define DEFAULT_BENCHMARK_TRIES 3
static int g_benchmark_tries = DEFAULT_BENCHMARK_TRIES;

/* Default max test size for quick sweep (well past any L3 cache)
 * 512MB is enough to measure main memory bandwidth on any system */
#define DEFAULT_MAX_TEST_SIZE (512UL * 1024 * 1024)  /* 512 MB */
static double g_max_runtime = DEFAULT_MAX_RUNTIME;

/* Adaptive latency tracking - used for early termination in series tests */
static double g_prev_latency_ns = 0;
static size_t g_prev_latency_size = 0;

/* Detected cache sizes (per core) */
static size_t g_l1_cache_size = 0;
static size_t g_l2_cache_size = 0;
static size_t g_l3_cache_size = 0;

/* Minimum total buffer size - adaptive based on cache topology */
static size_t g_min_total_size = 4096;  /* Default 4KB, updated after cache detection */

/* ============================================================================
 * Timing
 * ============================================================================ */

static inline double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ============================================================================
 * Memory operations
 * ============================================================================ */

/* Prevent compiler from optimizing away operations */
static volatile uint64_t g_sink = 0;

/* 
 * Memory operations - heavily optimized for bandwidth measurement
 * Key techniques:
 * 1. Multiple independent accumulators to break dependency chains
 * 2. Large unrolling (32 elements = 256 bytes per iteration)
 * 3. Force inlining to eliminate call overhead
 */

/* Read operation: XOR all 64-bit words with independent accumulators
 * XOR is faster than ADD and has no carry dependency chains */
static inline __attribute__((always_inline)) 
uint64_t mem_read(const void *buf, size_t size) {
    const uint64_t *p = (const uint64_t *)buf;
    const uint64_t *end = p + (size / sizeof(uint64_t));
    
    /* Use 8 independent accumulators - each one handles every 8th element */
    uint64_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;
    uint64_t x4 = 0, x5 = 0, x6 = 0, x7 = 0;
    
    /* Process 32 elements (256 bytes) per iteration */
    while (p + 32 <= end) {
        x0 ^= p[0];  x1 ^= p[1];  x2 ^= p[2];  x3 ^= p[3];
        x4 ^= p[4];  x5 ^= p[5];  x6 ^= p[6];  x7 ^= p[7];
        x0 ^= p[8];  x1 ^= p[9];  x2 ^= p[10]; x3 ^= p[11];
        x4 ^= p[12]; x5 ^= p[13]; x6 ^= p[14]; x7 ^= p[15];
        x0 ^= p[16]; x1 ^= p[17]; x2 ^= p[18]; x3 ^= p[19];
        x4 ^= p[20]; x5 ^= p[21]; x6 ^= p[22]; x7 ^= p[23];
        x0 ^= p[24]; x1 ^= p[25]; x2 ^= p[26]; x3 ^= p[27];
        x4 ^= p[28]; x5 ^= p[29]; x6 ^= p[30]; x7 ^= p[31];
        p += 32;
    }
    
    /* Handle remaining elements */
    while (p + 8 <= end) {
        x0 ^= p[0]; x1 ^= p[1]; x2 ^= p[2]; x3 ^= p[3];
        x4 ^= p[4]; x5 ^= p[5]; x6 ^= p[6]; x7 ^= p[7];
        p += 8;
    }
    while (p < end) {
        x0 ^= *p++;
    }
    
    return x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7;
}

/* Write operation: fill with pattern, heavily unrolled */
static inline __attribute__((always_inline))
void mem_write(void *buf, size_t size, uint64_t pattern) {
    uint64_t *p = (uint64_t *)buf;
    uint64_t *end = p + (size / sizeof(uint64_t));
    
    /* Process 32 elements (256 bytes) per iteration */
    while (p + 32 <= end) {
        p[0]  = pattern; p[1]  = pattern; p[2]  = pattern; p[3]  = pattern;
        p[4]  = pattern; p[5]  = pattern; p[6]  = pattern; p[7]  = pattern;
        p[8]  = pattern; p[9]  = pattern; p[10] = pattern; p[11] = pattern;
        p[12] = pattern; p[13] = pattern; p[14] = pattern; p[15] = pattern;
        p[16] = pattern; p[17] = pattern; p[18] = pattern; p[19] = pattern;
        p[20] = pattern; p[21] = pattern; p[22] = pattern; p[23] = pattern;
        p[24] = pattern; p[25] = pattern; p[26] = pattern; p[27] = pattern;
        p[28] = pattern; p[29] = pattern; p[30] = pattern; p[31] = pattern;
        p += 32;
    }
    
    /* Handle remaining */
    while (p < end) {
        *p++ = pattern;
    }
}

/* Copy operation: copy from src to dst, heavily unrolled */
static inline __attribute__((always_inline))
void mem_copy(void *dst, const void *src, size_t size) {
    const uint64_t *s = (const uint64_t *)src;
    uint64_t *d = (uint64_t *)dst;
    const uint64_t *end = s + (size / sizeof(uint64_t));
    
    /* Process 32 elements (256 bytes) per iteration */
    while (s + 32 <= end) {
        d[0]  = s[0];  d[1]  = s[1];  d[2]  = s[2];  d[3]  = s[3];
        d[4]  = s[4];  d[5]  = s[5];  d[6]  = s[6];  d[7]  = s[7];
        d[8]  = s[8];  d[9]  = s[9];  d[10] = s[10]; d[11] = s[11];
        d[12] = s[12]; d[13] = s[13]; d[14] = s[14]; d[15] = s[15];
        d[16] = s[16]; d[17] = s[17]; d[18] = s[18]; d[19] = s[19];
        d[20] = s[20]; d[21] = s[21]; d[22] = s[22]; d[23] = s[23];
        d[24] = s[24]; d[25] = s[25]; d[26] = s[26]; d[27] = s[27];
        d[28] = s[28]; d[29] = s[29]; d[30] = s[30]; d[31] = s[31];
        s += 32;
        d += 32;
    }
    
    /* Handle remaining */
    while (s < end) {
        *d++ = *s++;
    }
}

/* 
 * Memory latency test using pointer chasing
 * Each load depends on the previous one, preventing pipelining and prefetching
 */

/* Initialize pointer chain with random order to defeat prefetchers */
static void init_pointer_chain(void **buf, size_t count) {
    /* Start with sequential chain */
    for (size_t i = 0; i < count - 1; i++) {
        buf[i] = &buf[i + 1];
    }
    buf[count - 1] = &buf[0];  /* Make it circular */
    
    /* Fisher-Yates shuffle to randomize the chain */
    for (size_t i = count - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        /* Swap pointers */
        void *tmp = buf[i];
        buf[i] = buf[j];
        buf[j] = tmp;
    }
    
    /* Rebuild chain in shuffled order */
    for (size_t i = 0; i < count - 1; i++) {
        *(void **)buf[i] = buf[i + 1];
    }
    *(void **)buf[count - 1] = buf[0];
}

/* Chase pointers - each load depends on previous (true latency measurement) */
static inline __attribute__((always_inline))
void* mem_latency_chase(void *start, size_t accesses) {
    void **p = (void **)start;
    
    /* Unroll to reduce loop overhead but maintain dependency chain */
    size_t i = accesses;
    while (i >= 8) {
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        i -= 8;
    }
    while (i > 0) {
        p = (void **)*p;
        i--;
    }
    
    return p;
}

/* Adaptive latency measurement with early termination for series tests.
 * 
 * Strategy:
 * 1. First, traverse at least prev_size worth of accesses (warm up past known cache levels)
 * 2. Then measure in chunks, computing moving average latency
 * 3. Stop when latency has stabilized (plateau detected)
 * 
 * Returns: measured latency in nanoseconds, updates total_accesses with actual count
 */
static double mem_latency_adaptive(void *start, size_t buf_count, 
                                   size_t prev_size, double prev_latency_ns,
                                   size_t *total_accesses) {
    void **p = (void **)start;
    
    /* Minimum accesses: at least traverse past the previous buffer size */
    size_t min_accesses = prev_size / sizeof(void*);
    if (min_accesses < 1000) min_accesses = 1000;
    
    /* Chunk size for measurement (~100K accesses per chunk, ~10ms at 100ns latency) */
    size_t chunk_size = 100000;
    
    /* Maximum accesses (traverse entire buffer 3 times) */
    size_t max_accesses = buf_count * 3;
    if (max_accesses < min_accesses * 2) max_accesses = min_accesses * 2;
    
    /* Convergence parameters */
    #define LATENCY_WINDOW 5
    double recent_latencies[LATENCY_WINDOW] = {0};
    int window_idx = 0;
    int stable_count = 0;
    double converged_latency = 0;
    
    /* Do minimum accesses first (no timing, just warm up) */
    double warmup_start = get_time();
    size_t warmup_accesses = min_accesses;
    while (warmup_accesses >= 8) {
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        p = (void **)*p;
        warmup_accesses -= 8;
    }
    while (warmup_accesses > 0) {
        p = (void **)*p;
        warmup_accesses--;
    }
    double warmup_end = get_time();
    double warmup_latency = (warmup_end - warmup_start) * 1e9 / min_accesses;
    
    /* Initialize window with warmup latency */
    for (int i = 0; i < LATENCY_WINDOW; i++) {
        recent_latencies[i] = warmup_latency;
    }
    
    /* Now measure in chunks, checking for convergence */
    size_t accesses_done = min_accesses;
    
    while (accesses_done < max_accesses) {
        double chunk_start = get_time();
        
        /* Chase for one chunk */
        size_t remaining = chunk_size;
        while (remaining >= 8) {
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            p = (void **)*p;
            remaining -= 8;
        }
        while (remaining > 0) {
            p = (void **)*p;
            remaining--;
        }
        
        double chunk_end = get_time();
        double chunk_latency = (chunk_end - chunk_start) * 1e9 / chunk_size;
        accesses_done += chunk_size;
        
        /* Update sliding window */
        recent_latencies[window_idx] = chunk_latency;
        window_idx = (window_idx + 1) % LATENCY_WINDOW;
        
        /* Compute average and check for stability */
        double avg = 0, min_lat = 1e12, max_lat = 0;
        for (int i = 0; i < LATENCY_WINDOW; i++) {
            avg += recent_latencies[i];
            if (recent_latencies[i] < min_lat) min_lat = recent_latencies[i];
            if (recent_latencies[i] > max_lat) max_lat = recent_latencies[i];
        }
        avg /= LATENCY_WINDOW;
        
        /* Check if latency is stable (within 10% spread) and significantly
         * different from previous level (or we've done enough) */
        double spread = (max_lat - min_lat) / avg;
        int is_stable = (spread < 0.10);
        
        /* Also check if we've diverged from previous latency (new cache level) */
        int diverged_from_prev = (prev_latency_ns > 0 && avg > prev_latency_ns * 1.3);
        
        if (is_stable) {
            stable_count++;
            converged_latency = avg;
            
            /* Need 3 stable chunks AND either diverged or done enough */
            if (stable_count >= 3 && (diverged_from_prev || accesses_done >= min_accesses * 3)) {
                break;
            }
        } else {
            stable_count = 0;
        }
    }
    
    /* Store final pointer to prevent optimization */
    g_sink = (uint64_t)(uintptr_t)p;
    
    *total_accesses = accesses_done;
    return (converged_latency > 0) ? converged_latency : warmup_latency;
    
    #undef LATENCY_WINDOW
}

/* ============================================================================
 * Thread worker
 * ============================================================================ */

/* Single-threaded benchmark - no thread overhead */
static void run_single_thread(thread_work_t *work) {
    void *src = work->src;
    void *dst = work->dst;
    size_t size = work->size;
    uint64_t checksum = 0;
    int iterations = work->iterations;
    
    double start = get_time();
    
    switch (work->op) {
        case OP_READ:
            for (int i = 0; i < iterations; i++) {
                checksum += mem_read(src, size);
            }
            break;
            
        case OP_WRITE:
            for (int i = 0; i < iterations; i++) {
                mem_write(src, size, (uint64_t)i);
            }
            break;
            
        case OP_COPY:
            for (int i = 0; i < iterations; i++) {
                mem_copy(dst, src, size);
            }
            break;
            
        case OP_LATENCY: {
            /* Pointer chasing - count is number of pointers in the buffer */
            size_t count = size / sizeof(void*);
            size_t accesses = count * iterations;
            void *result = mem_latency_chase(src, accesses);
            checksum = (uint64_t)(uintptr_t)result;
            break;
        }
    }
    
    double end = get_time();
    
    work->elapsed = end - start;
    work->checksum = checksum;
}


static void* thread_worker(void *arg) {
    thread_work_t *work = (thread_work_t *)arg;
    void *src = work->src;
    void *dst = work->dst;
    size_t size = work->size;
    uint64_t checksum = 0;
    int iterations = work->iterations;
    
    /* Pin thread to specific CPU for consistent results.
     * This prevents the OS scheduler from moving threads between cores,
     * which causes huge variability in benchmark results.
     * 
     * On NUMA systems, threads are distributed evenly across nodes using
     * get_cpu_for_thread(), so cpu_id is already NUMA-balanced.
     * 
     * NOTE: We do NOT call numa_run_on_node() because it would OVERRIDE
     * the CPU pinning and allow the thread to run on any CPU in that node.
     * CPU pinning is more precise and gives better consistency. */
    int cpu = work->cpu_id;
    if (cpu >= 0 && cpu < g_num_cpus) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
    
#ifdef USE_NUMA
    /* Bind memory to the local NUMA node for this CPU.
     * This ensures memory is close to where it will be accessed. */
    if (numa_available() >= 0 && g_numa_nodes > 1 && cpu >= 0) {
        int node = numa_node_of_cpu(cpu);
        if (node >= 0) {
            /* Bind memory to the local node */
            unsigned long nodemask = 1UL << node;
            mbind(src, size, MPOL_BIND, &nodemask, g_numa_nodes + 1, MPOL_MF_MOVE);
            if (dst) mbind(dst, size, MPOL_BIND, &nodemask, g_numa_nodes + 1, MPOL_MF_MOVE);
        }
    }
#endif
    
    /* Initialize thread's private buffer (touch all pages) 
     * Note: For OP_LATENCY, pointer chain is already initialized in run_benchmark */
    if (work->op != OP_LATENCY) {
        memset(src, 0xAA, size);
        if (dst) memset(dst, 0, size);
    }
    
    /* Wait for all threads to be ready */
    pthread_barrier_wait(&g_barrier);
    
    double start = get_time();
    
    switch (work->op) {
        case OP_READ:
            for (int i = 0; i < iterations; i++) {
                checksum += mem_read(src, size);
            }
            break;
            
        case OP_WRITE:
            for (int i = 0; i < iterations; i++) {
                mem_write(src, size, (uint64_t)i);
            }
            break;
            
        case OP_COPY:
            for (int i = 0; i < iterations; i++) {
                mem_copy(dst, src, size);
            }
            break;
            
        case OP_LATENCY: {
            size_t count = size / sizeof(void*);
            size_t accesses = count * iterations;
            void *result = mem_latency_chase(src, accesses);
            checksum = (uint64_t)(uintptr_t)result;
            break;
        }
    }
    
    double end = get_time();
    
    work->elapsed = end - start;
    work->checksum = checksum;
    
    return NULL;
}

/* ============================================================================
 * Memory allocation
 * ============================================================================ */

static void* alloc_buffer(size_t size) {
    void *buf;
    
    /* Use mmap for large allocations, aligned and zero-initialized */
    buf = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (buf == MAP_FAILED) {
        return NULL;
    }
    
    /* Touch all pages to ensure they're allocated */
    memset(buf, 0, size);
    
    return buf;
}

static void free_buffer(void *buf, size_t size) {
    if (buf) {
        munmap(buf, size);
    }
}

/* ============================================================================
 * Cache topology detection using hwloc (portable: x86, arm64, etc.)
 * 
 * Install hwloc:
 *   Debian/Ubuntu: apt-get install libhwloc-dev
 *   RHEL/CentOS:   yum install hwloc-devel
 *   macOS:         brew install hwloc
 * ============================================================================ */

#ifdef USE_HWLOC
#include <hwloc.h>

static hwloc_topology_t g_topology = NULL;

/* Detect cache sizes using hwloc */
static void init_cache_info(void) {
    if (hwloc_topology_init(&g_topology) < 0) {
        goto use_defaults;
    }
    
    if (hwloc_topology_load(g_topology) < 0) {
        hwloc_topology_destroy(g_topology);
        g_topology = NULL;
        goto use_defaults;
    }
    
    /* Find cache sizes by iterating through cache objects */
    int depth;
    
    /* L1 Data Cache */
    depth = hwloc_get_type_depth(g_topology, HWLOC_OBJ_L1CACHE);
    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(g_topology, depth, 0);
        if (obj && obj->attr && obj->attr->cache.type != HWLOC_OBJ_CACHE_INSTRUCTION) {
            g_l1_cache_size = obj->attr->cache.size;
        }
    }
    
    /* L2 Cache */
    depth = hwloc_get_type_depth(g_topology, HWLOC_OBJ_L2CACHE);
    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(g_topology, depth, 0);
        if (obj && obj->attr) {
            g_l2_cache_size = obj->attr->cache.size;
        }
    }
    
    /* L3 Cache */
    depth = hwloc_get_type_depth(g_topology, HWLOC_OBJ_L3CACHE);
    if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
        hwloc_obj_t obj = hwloc_get_obj_by_depth(g_topology, depth, 0);
        if (obj && obj->attr) {
            g_l3_cache_size = obj->attr->cache.size;
        }
    }
    
    /* Count total L3 cache (sum across all L3 objects for distributed caches) */
    if (g_l3_cache_size > 0) {
        depth = hwloc_get_type_depth(g_topology, HWLOC_OBJ_L3CACHE);
        int num_l3 = hwloc_get_nbobjs_by_depth(g_topology, depth);
        if (g_verbose && num_l3 > 1) {
            fprintf(stderr, "Note: %d L3 caches detected (distributed across dies)\n", num_l3);
        }
    }
    
use_defaults:
    /* Set defaults if detection failed */
    if (g_l1_cache_size == 0) g_l1_cache_size = 32 * 1024;      /* 32 KB */
    if (g_l2_cache_size == 0) g_l2_cache_size = 256 * 1024;     /* 256 KB */
    if (g_l3_cache_size == 0) g_l3_cache_size = 8 * 1024 * 1024; /* 8 MB */
    
    /* Calculate adaptive minimum total size:
     * Use 16KB per thread × num_cpus so each thread has a reliable buffer size.
     * This ensures all CPUs can participate with meaningful measurements. */
    g_min_total_size = 16384 * g_num_cpus;  /* 16KB per thread minimum */
    
    if (g_verbose) {
        fprintf(stderr, "Cache (hwloc): L1d=%zuKB, L2=%zuKB, L3=%zuKB (per core)\n",
                g_l1_cache_size / 1024, g_l2_cache_size / 1024, g_l3_cache_size / 1024);
        fprintf(stderr, "Minimum total test size: %zu KB (16KB × %d CPUs)\n",
                g_min_total_size / 1024, g_num_cpus);
    }
}

static void cleanup_hwloc(void) {
    if (g_topology) {
        hwloc_topology_destroy(g_topology);
        g_topology = NULL;
    }
}

#else /* !USE_HWLOC - fallback to sysfs parsing */

/* Parse cache size from sysfs (handles "48K", "1024K", "32768K" format) */
static size_t parse_cache_size_sysfs(const char *str) {
    size_t size = 0;
    char unit = 0;
    if (sscanf(str, "%zu%c", &size, &unit) >= 1) {
        if (unit == 'K' || unit == 'k') size *= 1024;
        else if (unit == 'M' || unit == 'm') size *= 1024 * 1024;
    }
    return size;
}

/* Read cache info from sysfs */
static void init_cache_info(void) {
    char path[256];
    char buf[64];
    FILE *f;
    
    /* Try to read cache info from sysfs (Linux) */
    for (int index = 0; index < 10; index++) {
        /* Read level */
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/level", index);
        f = fopen(path, "r");
        if (!f) continue;
        int level = -1;
        if (fgets(buf, sizeof(buf), f)) level = atoi(buf);
        fclose(f);
        if (level < 0) continue;
        
        /* Read type */
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/type", index);
        f = fopen(path, "r");
        if (!f) continue;
        char type[32] = "";
        if (fgets(type, sizeof(type), f)) type[strcspn(type, "\n")] = 0;
        fclose(f);
        
        /* Skip instruction caches */
        if (strcmp(type, "Instruction") == 0) continue;
        
        /* Read size */
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
        f = fopen(path, "r");
        if (!f) continue;
        size_t size = 0;
        if (fgets(buf, sizeof(buf), f)) size = parse_cache_size_sysfs(buf);
        fclose(f);
        
        if (size == 0) continue;
        
        switch (level) {
            case 1: if (g_l1_cache_size == 0) g_l1_cache_size = size; break;
            case 2: if (g_l2_cache_size == 0) g_l2_cache_size = size; break;
            case 3: if (g_l3_cache_size == 0) g_l3_cache_size = size; break;
        }
    }
    
    /* Set defaults if detection failed */
    if (g_l1_cache_size == 0) g_l1_cache_size = 32 * 1024;      /* 32 KB */
    if (g_l2_cache_size == 0) g_l2_cache_size = 256 * 1024;     /* 256 KB */
    if (g_l3_cache_size == 0) g_l3_cache_size = 8 * 1024 * 1024; /* 8 MB */
    
    /* Calculate adaptive minimum total size:
     * Use 16KB per thread × num_cpus so each thread has a reliable buffer size. */
    g_min_total_size = 16384 * g_num_cpus;  /* 16KB per thread minimum */
    
    if (g_verbose) {
        fprintf(stderr, "Cache (sysfs): L1d=%zuKB, L2=%zuKB, L3=%zuKB (per core)\n",
                g_l1_cache_size / 1024, g_l2_cache_size / 1024, g_l3_cache_size / 1024);
        fprintf(stderr, "Minimum total test size: %zu KB (16KB × %d CPUs)\n",
                g_min_total_size / 1024, g_num_cpus);
    }
}

static void cleanup_hwloc(void) {
    /* No-op when hwloc is not used */
}

#endif /* USE_HWLOC */

/* ============================================================================
 * NUMA support
 * ============================================================================ */

static void init_numa_topology(void) {
    /* Initialize topology arrays */
    memset(g_cpus_per_node, 0, sizeof(g_cpus_per_node));
    memset(g_node_cpus, 0, sizeof(g_node_cpus));
    
#ifdef USE_NUMA
    if (numa_available() >= 0 && g_numa_nodes > 1) {
        /* Build CPU-to-node mapping using libnuma */
        for (int cpu = 0; cpu < g_num_cpus && cpu < MAX_NUMA_NODES * MAX_CPUS_PER_NODE; cpu++) {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0 && node < MAX_NUMA_NODES) {
                int idx = g_cpus_per_node[node];
                if (idx < MAX_CPUS_PER_NODE) {
                    g_node_cpus[node][idx] = cpu;
                    g_cpus_per_node[node]++;
                }
            }
        }
        
        if (g_verbose) {
            fprintf(stderr, "NUMA topology:\n");
            for (int node = 0; node < g_numa_nodes; node++) {
                fprintf(stderr, "  Node %d: %d CPUs (first: %d, last: %d)\n",
                        node, g_cpus_per_node[node],
                        g_cpus_per_node[node] > 0 ? g_node_cpus[node][0] : -1,
                        g_cpus_per_node[node] > 0 ? g_node_cpus[node][g_cpus_per_node[node]-1] : -1);
            }
        }
    } else
#endif
    {
        /* UMA or NUMA not enabled: all CPUs on "node 0" */
        for (int cpu = 0; cpu < g_num_cpus && cpu < MAX_CPUS_PER_NODE; cpu++) {
            g_node_cpus[0][cpu] = cpu;
        }
        g_cpus_per_node[0] = g_num_cpus < MAX_CPUS_PER_NODE ? g_num_cpus : MAX_CPUS_PER_NODE;
    }
}

/* Get CPU for thread i, distributing evenly across NUMA nodes */
static int get_cpu_for_thread(int thread_id, int num_threads) {
    if (g_numa_nodes <= 1 || num_threads <= 1) {
        /* UMA or single thread: use sequential assignment */
        return thread_id % g_num_cpus;
    }
    
    /* NUMA: distribute threads round-robin across nodes */
    int node = thread_id % g_numa_nodes;
    int thread_on_node = thread_id / g_numa_nodes;
    
    if (g_cpus_per_node[node] == 0) {
        /* Fallback if node has no CPUs (shouldn't happen) */
        return thread_id % g_num_cpus;
    }
    
    int cpu_idx = thread_on_node % g_cpus_per_node[node];
    return g_node_cpus[node][cpu_idx];
}

static void init_numa(void) {
#ifdef USE_NUMA
    if (numa_available() >= 0) {
        g_numa_nodes = numa_max_node() + 1;
        if (g_verbose) {
            fprintf(stderr, "NUMA: %d nodes detected (libnuma enabled)\n", g_numa_nodes);
        }
    } else {
        g_numa_nodes = 1;
        if (g_verbose) {
            fprintf(stderr, "NUMA: not available (libnuma enabled but no NUMA support)\n");
        }
    }
#else
    g_numa_nodes = 1;
    if (g_verbose) {
        fprintf(stderr, "NUMA: disabled (compile with -DUSE_NUMA -lnuma to enable)\n");
    }
#endif
    
    /* Build NUMA topology after detecting nodes */
    init_numa_topology();
}


/* ============================================================================
 * System info
 * ============================================================================ */

static void init_system_info(void) {
    /* Get number of CPUs */
    g_num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (g_num_cpus < 1) g_num_cpus = 1;
    
    /* Get total memory */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        g_total_memory = (size_t)pages * (size_t)page_size;
    } else {
        g_total_memory = 1024UL * 1024 * 1024;  /* Default 1GB */
    }
    
    if (g_verbose) {
        fprintf(stderr, "System: %d CPUs, %.2f GB memory\n", 
                g_num_cpus, g_total_memory / (1024.0 * 1024 * 1024));
    }
    
    /* Detect cache topology (must be called after g_num_cpus is set) */
    init_cache_info();
}

/* ============================================================================
 * Benchmark runner
 * ============================================================================ */

/* Calibrate iterations for target time */
static int calibrate_iterations(void *src, void *dst, size_t size, 
                                operation_t op) {
    /* Warmup pass */
    switch (op) {
        case OP_READ:
            g_sink += mem_read(src, size);
            break;
        case OP_WRITE:
            mem_write(src, size, 0x1234567890ABCDEFULL);
            break;
        case OP_COPY:
            mem_copy(dst, src, size);
            break;
        case OP_LATENCY: {
            size_t count = size / sizeof(void*);
            g_sink += (uint64_t)(uintptr_t)mem_latency_chase(src, count);
            break;
        }
    }
    
    /* Calibration run - do multiple passes for small buffers */
    int cal_iters = (size < 65536) ? 100 : 10;
    
    double start = get_time();
    for (int i = 0; i < cal_iters; i++) {
        switch (op) {
            case OP_READ:
                g_sink += mem_read(src, size);
                break;
            case OP_WRITE:
                mem_write(src, size, 0x1234567890ABCDEFULL);
                break;
            case OP_COPY:
                mem_copy(dst, src, size);
                break;
            case OP_LATENCY: {
                size_t count = size / sizeof(void*);
                g_sink += (uint64_t)(uintptr_t)mem_latency_chase(src, count);
                break;
            }
        }
    }
    double elapsed = get_time() - start;
    double time_per_iter = elapsed / cal_iters;
    
    if (time_per_iter < 1e-9) time_per_iter = 1e-9;
    
    /* Estimate iterations needed for target time */
    int iters = (int)(TARGET_TIME_PER_TEST / time_per_iter);
    
    /* Clamp to valid range */
    if (iters < MIN_ITERATIONS) iters = MIN_ITERATIONS;
    if (iters > MAX_ITERATIONS) iters = MAX_ITERATIONS;
    
    return iters;
}


/* Run a single benchmark
 * Each thread gets its own buffer (like bw_mem) to measure aggregate system bandwidth
 */
static result_t run_benchmark(size_t size, operation_t op, int nthreads) {
    result_t result = {0};
    result.size = size;
    result.op = op;
    result.threads = nthreads;
    
    /* For single-threaded, use simple path */
    if (nthreads == 1) {
        void *src = alloc_buffer(size);
        void *dst = (op == OP_COPY) ? alloc_buffer(size) : NULL;
        
        if (!src || (op == OP_COPY && !dst)) {
            free_buffer(src, size);
            free_buffer(dst, size);
            return result;
        }
        
        if (op == OP_LATENCY) {
            /* Initialize pointer chain for latency test */
            size_t count = size / sizeof(void*);
            if (count < 2) count = 2;
            init_pointer_chain((void**)src, count);
        } else {
            memset(src, 0xAA, size);
            if (dst) memset(dst, 0, size);
        }
        
        int iterations = calibrate_iterations(src, dst, size, op);
        result.iterations = iterations;
        
        thread_work_t work = {0};
        work.src = src;
        work.dst = dst;
        work.size = size;
        work.op = op;
        work.iterations = iterations;
        
        run_single_thread(&work);
        
        g_sink += work.checksum;
        result.elapsed_s = work.elapsed;
        
        if (op == OP_LATENCY) {
            /* Latency = time / total_accesses (in nanoseconds) */
            size_t count = size / sizeof(void*);
            size_t total_accesses = count * iterations;
            if (work.elapsed > 0 && total_accesses > 0) {
                result.latency_ns = (work.elapsed * 1e9) / total_accesses;
            }
        } else {
            /* Bandwidth = (size per thread * threads * iterations) / time
             * Note: for copy, we report buffer size (not 2x) to match bw_mem */
            size_t bytes_transferred = size * iterations;
            
            if (work.elapsed > 0) {
                result.bandwidth_mb_s = (bytes_transferred / (1024.0 * 1024.0)) / work.elapsed;
            }
        }
        
        free_buffer(src, size);
        free_buffer(dst, size);
        return result;
    }
    
    /* Multi-threaded: each thread gets its own buffer */
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    thread_work_t *work = malloc(nthreads * sizeof(thread_work_t));
    void **src_bufs = calloc(nthreads, sizeof(void*));
    void **dst_bufs = calloc(nthreads, sizeof(void*));
    
    if (!threads || !work || !src_bufs || !dst_bufs) {
        free(threads);
        free(work);
        free(src_bufs);
        free(dst_bufs);
        return result;
    }
    
    /* Allocate per-thread buffers */
    for (int i = 0; i < nthreads; i++) {
        src_bufs[i] = alloc_buffer(size);
        dst_bufs[i] = (op == OP_COPY) ? alloc_buffer(size) : NULL;
        
        if (!src_bufs[i] || (op == OP_COPY && !dst_bufs[i])) {
            /* Cleanup on failure */
            for (int j = 0; j <= i; j++) {
                free_buffer(src_bufs[j], size);
                free_buffer(dst_bufs[j], size);
            }
            free(threads);
            free(work);
            free(src_bufs);
            free(dst_bufs);
            if (g_verbose) {
                fprintf(stderr, "Failed to allocate %zu bytes × %d threads\n", size, nthreads);
            }
            return result;
        }
    }
    
    /* Initialize buffers */
    if (op == OP_LATENCY) {
        /* Initialize pointer chains for all buffers */
        size_t count = size / sizeof(void*);
        if (count < 2) count = 2;
        for (int i = 0; i < nthreads; i++) {
            init_pointer_chain((void**)src_bufs[i], count);
        }
    } else {
        /* Initialize first buffer for calibration */
        memset(src_bufs[0], 0xAA, size);
        if (dst_bufs[0]) memset(dst_bufs[0], 0, size);
    }
    
    int iterations = calibrate_iterations(src_bufs[0], dst_bufs[0], size, op);
    result.iterations = iterations;
    
    pthread_barrier_init(&g_barrier, NULL, nthreads);
    
    /* Setup per-thread work with NUMA-balanced CPU assignment */
    for (int i = 0; i < nthreads; i++) {
        work[i].src = src_bufs[i];
        work[i].dst = dst_bufs[i];
        work[i].size = size;
        work[i].op = op;
        work[i].iterations = iterations;
        work[i].checksum = 0;
        work[i].elapsed = 0;
        work[i].thread_id = i;
        work[i].cpu_id = get_cpu_for_thread(i, nthreads);  /* NUMA-balanced CPU assignment */
        work[i].numa_node = -1;  /* Not used - memory binding done in thread using cpu_id */
    }
    
    /* Launch threads */
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&threads[i], NULL, thread_worker, &work[i]);
    }
    
    /* Wait for completion */
    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Find max elapsed time (determines overall bandwidth) */
    double max_elapsed = 0;
    uint64_t total_checksum = 0;
    for (int i = 0; i < nthreads; i++) {
        if (work[i].elapsed > max_elapsed) {
            max_elapsed = work[i].elapsed;
        }
        total_checksum += work[i].checksum;
    }
    
    g_sink += total_checksum;
    result.elapsed_s = max_elapsed;
    
    if (op == OP_LATENCY) {
        /* Latency = time / total_accesses (in nanoseconds)
         * For latency, we report average per-thread latency, not aggregate */
        size_t count = size / sizeof(void*);
        size_t accesses_per_thread = count * iterations;
        if (max_elapsed > 0 && accesses_per_thread > 0) {
            result.latency_ns = (max_elapsed * 1e9) / accesses_per_thread;
        }
    } else {
        /* Bandwidth = (size per thread * threads * iterations) / time 
         * This gives aggregate bandwidth across all threads
         * Note: for copy, we report buffer size (not 2x) to match bw_mem */
        size_t bytes_transferred = (size_t)size * nthreads * iterations;
        
        if (max_elapsed > 0) {
            result.bandwidth_mb_s = (bytes_transferred / (1024.0 * 1024.0)) / max_elapsed;
        }
    }
    
    /* Cleanup */
    pthread_barrier_destroy(&g_barrier);
    for (int i = 0; i < nthreads; i++) {
        free_buffer(src_bufs[i], size);
        free_buffer(dst_bufs[i], size);
    }
    free(threads);
    free(work);
    free(src_bufs);
    free(dst_bufs);
    
    return result;
}

/* Run benchmark multiple times and return best result (like lmbench TRIES)
 * For bandwidth: best = highest bandwidth
 * For latency: best = lowest latency
 * 
 * First run is a warmup (discarded) to allow CPU frequency to ramp up
 * and caches to warm. This dramatically reduces result variability.
 */
static result_t run_benchmark_best(size_t size, operation_t op, int nthreads) {
    result_t best = {0};
    
    /* Warmup run - discarded.
     * This allows: CPU to reach turbo frequency, caches to warm,
     * thread scheduling to stabilize. Critical for consistent results. */
    (void)run_benchmark(size, op, nthreads);
    
    for (int try = 0; try < g_benchmark_tries; try++) {
        result_t r = run_benchmark(size, op, nthreads);
        
        if (try == 0) {
            best = r;
        } else {
            if (op == OP_LATENCY) {
                /* For latency: lower is better */
                if (r.latency_ns > 0 && r.latency_ns < best.latency_ns) {
                    best = r;
                }
            } else {
                /* For bandwidth: higher is better */
                if (r.bandwidth_mb_s > best.bandwidth_mb_s) {
                    best = r;
                }
            }
        }
    }
    
    return best;
}

/* ============================================================================
 * Main benchmark loop
 * ============================================================================ */

/* Generate thread counts dynamically based on CPU count
 * 
 * Strategy:
 * - Always include 1 (single-threaded baseline)
 * - Powers of 2 up to nproc
 * - Always include nproc itself
 * - Include nproc * 1.5 and nproc * 2 to test oversubscription
 * 
 * Examples:
 *   32 cores:   1, 2, 4, 8, 16, 32, 48, 64           (8 values)
 *   192 cores:  1, 2, 4, 8, 16, 32, 64, 128, 192, 288, 384  (11 values)
 *   1920 cores: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1920, 2880, 3840 (14 values)
 */
static int* get_thread_counts(int *count) {
    int nproc = g_num_cpus;
    if (nproc < 1) nproc = 1;
    
    /* Cap at nproc - oversubscription causes unreliable benchmark results
     * due to context switching, cache thrashing, and scheduler interference */
    int max_threads = nproc;
    
    /* Allocate more than enough space */
    int *tc = malloc(32 * sizeof(int));
    int n = 0;
    
    /* Add powers of 2 up to nproc */
    for (int t = 1; t <= max_threads; t *= 2) {
        tc[n++] = t;
    }
    
    /* Add nproc if not already in list (i.e., not a power of 2) */
    if (tc[n-1] != nproc) {
        tc[n++] = nproc;
    }
    
    tc[n] = 0;  /* Sentinel */
    *count = n;
    return tc;
}

/* Get sizes to test */
static size_t* get_sizes(int *count) {
    /* Limit to available memory (use max 50% of RAM) */
    size_t max_size = g_total_memory / 2;
    
    /* Start from minimum total size (adaptive based on cache × CPUs) */
    size_t min_size = g_min_total_size;
    
    int n = 0;
    for (int i = 0; DEFAULT_SIZES[i] != 0; i++) {
        if (DEFAULT_SIZES[i] >= min_size && DEFAULT_SIZES[i] <= max_size) n++;
    }
    
    /* Always have at least one size */
    if (n == 0) n = 1;
    
    size_t *sizes = malloc((n + 1) * sizeof(size_t));
    int j = 0;
    for (int i = 0; DEFAULT_SIZES[i] != 0; i++) {
        if (DEFAULT_SIZES[i] >= min_size && DEFAULT_SIZES[i] <= max_size) {
            sizes[j++] = DEFAULT_SIZES[i];
        }
    }
    
    /* If no sizes matched, use minimum size */
    if (j == 0) {
        sizes[j++] = min_size;
    }
    
    sizes[j] = 0;
    *count = j;
    return sizes;
}

static void print_csv_header(void) {
    printf("size_kb,operation,bandwidth_mb_s,latency_ns,threads,iterations,elapsed_s\n");
}

static void print_result(const result_t *r) {
    size_t size_kb = r->size / 1024;
    if (r->op == OP_LATENCY) {
        /* For latency test, bandwidth is 0, latency has value */
        printf("%zu,%s,0,%.2f,%d,%d,%.6f\n",
               size_kb, OP_NAMES[r->op], r->latency_ns,
               r->threads, r->iterations, r->elapsed_s);
    } else {
        /* For bandwidth tests, latency is 0 */
        printf("%zu,%s,%.2f,0,%d,%d,%.6f\n",
               size_kb, OP_NAMES[r->op], r->bandwidth_mb_s,
               r->threads, r->iterations, r->elapsed_s);
    }
}

/* Minimum per-thread buffer size for reliable measurements.
 * 
 * Very small buffers (< 16KB) cause unreliable results because:
 * - Timing overhead dominates the measurement
 * - Data may stay in registers instead of cache/memory
 * - Loop overhead becomes significant
 * 
 * We use 16KB as the minimum, which:
 * - Is large enough for stable timing
 * - Ensures we're measuring actual memory/cache performance
 * - Matches bw_mem's typical minimum test size
 */
static size_t get_min_per_thread_size(void) {
    /* 16KB minimum for reliable measurements */
    return 16384;
}

/* Maximum buffer size for latency test.
 * Must exceed largest L3 caches to measure true DRAM latency.
 * AMD EPYC 9754 (Genoa-X) has 1.1GB L3 cache, so we need > 1.1GB.
 * 2GB should cover any current processor. */
#define MAX_LATENCY_SIZE (2UL * 1024 * 1024 * 1024)  /* 2 GB */

/* Find best thread/buffer configuration for a given TOTAL memory size.
 * 
 * For total_size = 1MB, we try:
 *   1 thread  × 1MB buffer   = 1MB total
 *   2 threads × 512KB buffer = 1MB total
 *   4 threads × 256KB buffer = 1MB total
 *   etc.
 * 
 * For COPY operations, total_size includes BOTH src and dst buffers:
 *   1 thread  × 512KB src + 512KB dst = 1MB total
 *   2 threads × 256KB src + 256KB dst = 1MB total
 * 
 * This ensures size_kb in output represents actual total memory footprint
 * consistently across all operations.
 */
static result_t find_best_config(size_t total_size, operation_t op, 
                                 int *thread_counts, int tc_count) {
    result_t best = {0};
    best.size = total_size;
    best.op = op;
    
    /* For latency test: single-thread, cap buffer size.
     * Need buffer larger than L3 cache to measure true DRAM latency.
     * Cap at MAX_LATENCY_SIZE (2GB) or 25% of available memory, whichever is smaller. */
    if (op == OP_LATENCY) {
        size_t max_latency = MAX_LATENCY_SIZE;
        if (g_total_memory / 4 < max_latency) {
            max_latency = g_total_memory / 4;
        }
        size_t latency_size = (total_size > max_latency) ? max_latency : total_size;
        
        /* Use adaptive early termination for series tests (not single-size mode)
         * when:
         * - We have previous latency data
         * - Current size is larger than previous
         * - Current size is > 1.5GB (safely past even AMD EPYC 9754's 1.1GB L3)
         * - Previous latency indicates we're already in DRAM territory (> 50ns)
         *   Note: L3 latency is typically 10-40ns, DRAM is 60-120ns
         */
        int in_dram_region = (g_prev_latency_ns > 50.0);
        int past_l3_cache = (latency_size > 1536UL * 1024 * 1024);  /* 1.5 GB */
        
        if (g_single_size == 0 && g_prev_latency_ns > 0 && 
            latency_size > g_prev_latency_size && past_l3_cache && in_dram_region) {
            /* Adaptive latency measurement */
            void *buf = alloc_buffer(latency_size);
            if (buf) {
                size_t count = latency_size / sizeof(void*);
                if (count < 2) count = 2;
                init_pointer_chain((void**)buf, count);
                
                size_t actual_accesses = 0;
                double start = get_time();
                double latency = mem_latency_adaptive(buf, count,
                                                      g_prev_latency_size, g_prev_latency_ns,
                                                      &actual_accesses);
                double elapsed = get_time() - start;
                
                free_buffer(buf, latency_size);
                
                best.size = total_size;
                best.op = op;
                best.threads = 1;
                best.latency_ns = latency;
                best.elapsed_s = elapsed;
                best.iterations = (int)(actual_accesses / count);
                if (best.iterations < 1) best.iterations = 1;
                
                /* Update tracking for next test */
                g_prev_latency_ns = latency;
                g_prev_latency_size = latency_size;
                
                return best;
            }
        }
        
        /* Standard latency measurement (for first test or single-size mode) */
        best = run_benchmark_best(latency_size, op, 1);
        best.size = total_size;  /* Report requested size, not capped size */
        
        /* Update tracking for next test */
        g_prev_latency_ns = best.latency_ns;
        g_prev_latency_size = latency_size;
        
        return best;
    }
    
    /* For copy: total_size is split between src and dst buffers
     * So effective_size is what we divide among threads for each buffer */
    int bufs_per_op = (op == OP_COPY) ? 2 : 1;
    size_t effective_size = total_size / bufs_per_op;
    
    /* Try different thread counts where total memory stays constant */
    for (int i = 0; i < tc_count; i++) {
        int nthreads = thread_counts[i];
        if (nthreads < 1) continue;
        
        /* Calculate per-thread buffer size */
        size_t per_thread_size = effective_size / nthreads;
        
        /* Skip if per-thread buffer would be too small */
        if (per_thread_size < get_min_per_thread_size()) {
            continue;
        }
        
        /* Skip if doesn't divide evenly (to keep total exact) */
        if (per_thread_size * nthreads != effective_size) {
            continue;
        }
        
        /* Check memory limit */
        size_t memory_needed = per_thread_size * nthreads * bufs_per_op;
        if (memory_needed > g_total_memory / 4) {
            continue;
        }
        
        /* Run benchmark with this configuration */
        result_t r = run_benchmark_best(per_thread_size, op, nthreads);
        
        /* Store total size (not per-thread) for reporting */
        r.size = total_size;
        
        if (r.bandwidth_mb_s > best.bandwidth_mb_s) {
            best = r;
        }
    }
    
    /* If no valid configuration found, try single thread */
    if (best.bandwidth_mb_s == 0) {
        size_t single_thread_size = effective_size;
        best = run_benchmark_best(single_thread_size, op, 1);
        best.size = total_size;
    }
    
    return best;
}

static void run_all_benchmarks(void) {
    double start_time = get_time();
    
    /* Reset adaptive latency tracking for this run */
    g_prev_latency_ns = 0;
    g_prev_latency_size = 0;
    
    /* Get thread counts */
    int tc_count;
    int *thread_counts = get_thread_counts(&tc_count);
    
    /* Single size mode: test only the specified size */
    if (g_single_size > 0) {
        if (g_verbose) {
            fprintf(stderr, "Testing total size: %zu KB with various thread/buffer configs\n",
                    g_single_size / 1024);
        }
        
        print_csv_header();
        
        for (int op = 0; op < 4 && g_running; op++) {
            result_t best = find_best_config(g_single_size, (operation_t)op, 
                                            thread_counts, tc_count);
            
            if (best.bandwidth_mb_s > 0 || best.latency_ns > 0) {
                print_result(&best);
                fflush(stdout);
            }
        }
        
        free(thread_counts);
        
        if (g_verbose) {
            double total = get_time() - start_time;
            fprintf(stderr, "Total runtime: %.1f seconds\n", total);
        }
        return;
    }
    
    /* Normal mode: test all sizes */
    int size_count;
    size_t *sizes = get_sizes(&size_count);
    
    /* Determine max test size */
    size_t max_test_size = g_full_sweep ? sizes[size_count-1] : DEFAULT_MAX_TEST_SIZE;
    /* Don't exceed available memory */
    if (max_test_size > sizes[size_count-1]) {
        max_test_size = sizes[size_count-1];
    }
    
    if (g_verbose) {
        fprintf(stderr, "Testing %d sizes, up to %d thread configurations\n",
                size_count, tc_count);
        fprintf(stderr, "Max test size: %.2f MB%s\n", 
                max_test_size / (1024.0 * 1024.0),
                g_full_sweep ? " (full sweep)" : " (use -f for full sweep)");
        if (g_max_runtime > 0) {
            fprintf(stderr, "Target runtime: %.0f seconds\n", g_max_runtime);
        } else {
            fprintf(stderr, "Target runtime: unlimited\n");
        }
    }
    
    print_csv_header();
    
    /* Run benchmarks from small to large */
    for (int s = 0; s < size_count && g_running; s++) {
        size_t size = sizes[s];
        
        /* Skip sizes beyond max test size */
        if (size > max_test_size) {
            continue;
        }
        
        for (int op = 0; op < 4 && g_running; op++) {
            result_t best = find_best_config(size, (operation_t)op, 
                                             thread_counts, tc_count);
            
            if (best.bandwidth_mb_s > 0 || best.latency_ns > 0) {
                print_result(&best);
                fflush(stdout);
            }
            
            /* Check time budget (0 = unlimited) */
            if (g_max_runtime > 0) {
                double elapsed = get_time() - start_time;
                if (elapsed > g_max_runtime) {
                    if (g_verbose) {
                        fprintf(stderr, "Time limit reached (%.1f s)\n", elapsed);
                    }
                    g_running = 0;
                    break;
                }
            }
        }
    }
    
    free(sizes);
    free(thread_counts);
    
    if (g_verbose) {
        double total = get_time() - start_time;
        fprintf(stderr, "Total runtime: %.1f seconds\n", total);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void usage(const char *prog) {
    fprintf(stderr, "sc-membench %s - Memory Bandwidth Benchmark\n\n", VERSION);
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h          Show this help\n");
    fprintf(stderr, "  -v          Verbose output (to stderr)\n");
    fprintf(stderr, "  -s SIZE_KB  Test only this size (in KB), e.g. -s 1024 for 1MB\n");
    fprintf(stderr, "  -f          Full sweep (test all sizes up to 50%% RAM)\n");
    fprintf(stderr, "              Default: test up to 512 MB (enough for main memory BW)\n");
    fprintf(stderr, "  -t SECONDS  Maximum runtime, 0 = unlimited (default: unlimited)\n");
    fprintf(stderr, "  -r TRIES    Repeat each test N times, report best (default: %d)\n", DEFAULT_BENCHMARK_TRIES);
    fprintf(stderr, "\n");
    fprintf(stderr, "Output: CSV to stdout with columns:\n");
    fprintf(stderr, "  size_kb        - Memory size tested (KB)\n");
    fprintf(stderr, "  operation      - read, write, copy, or latency\n");
    fprintf(stderr, "  bandwidth_mb_s - Bandwidth in MB/s (0 for latency test)\n");
    fprintf(stderr, "  latency_ns     - Memory latency in nanoseconds (0 for bandwidth tests)\n");
    fprintf(stderr, "  threads        - Thread count for best result\n");
    fprintf(stderr, "  iterations     - Iterations performed\n");
    fprintf(stderr, "  elapsed_s      - Elapsed time in seconds\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "The latency test uses pointer chasing to measure true memory\n");
    fprintf(stderr, "access latency without pipelining or prefetching effects.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Compile with -DUSE_NUMA -lnuma for NUMA support.\n");
}

int main(int argc, char *argv[]) {
    int opt;
    
    while ((opt = getopt(argc, argv, "hvfs:t:r:")) != -1) {
        switch (opt) {
            case 'h':
                usage(argv[0]);
                return 0;
            case 'v':
                g_verbose = 1;
                break;
            case 'f':
                g_full_sweep = 1;
                break;
            case 'r':
                g_benchmark_tries = atoi(optarg);
                if (g_benchmark_tries < 1) g_benchmark_tries = 1;
                break;
            case 's': {
                long size_kb = atol(optarg);
                if (size_kb <= 0) {
                    fprintf(stderr, "Invalid size: %s\n", optarg);
                    return 1;
                }
                g_single_size = (size_t)size_kb * 1024;  /* Convert KB to bytes */
                break;
            }
            case 't':
                g_max_runtime = atof(optarg);
                if (g_max_runtime < 0) {
                    fprintf(stderr, "Invalid runtime: %s (use 0 for unlimited)\n", optarg);
                    return 1;
                }
                break;
            default:
                usage(argv[0]);
                return 1;
        }
    }
    
    /* Initialize */
    srand((unsigned int)time(NULL));  /* Seed RNG for pointer chain randomization */
    init_system_info();
    init_numa();
    
    /* Run benchmarks */
    run_all_benchmarks();
    
    /* Cleanup */
    cleanup_hwloc();
    
    return 0;
}

