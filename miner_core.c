/*
 * miner_core.c v3 — True parallel Bitcoin SHA-256d mining
 *
 * v3 changes vs v2:
 *   - Py_BEGIN/END_ALLOW_THREADS: releases the GIL during the hot loop so
 *     all 8 threads run on real CPU cores simultaneously (v2 was single-threaded
 *     despite launching 8 threads — the GIL serialized them all)
 *   - ARM SHA-256 hardware intrinsics path (vsha256h/vsha256h2/su0/su1):
 *     uses the phone's dedicated SHA-256 silicon when compiled with
 *     -march=armv8-a+sha2 — roughly 4-8x faster than generic C SHA-256
 *   - Generic C fallback for non-ARM or older devices
 *
 * Combined effect of GIL fix + ARM SHA2 on 8-core ARM64: expected 16-30x
 * speedup over v1 (OpenSSL, 4 threads, GIL held).
 *
 * API unchanged:
 *   mine_range(first_block, second_prefix, target_le, soft_target_le,
 *              nonce_start, nonce_end)
 *     -> (winning_nonce | None, hashes_done, soft_shares)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdint.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SHA2)
#  include <arm_neon.h>
#  include <sys/auxv.h>
#  ifndef HWCAP_SHA2
#    define HWCAP_SHA2 (1UL << 6)
#  endif
#  define USE_ARM_SHA2 1
#endif

/* ── SHA-256 round constants ──────────────────────────────────────────────── */

static const uint32_t K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

/* SHA-256 initial hash values (H0) */
static const uint32_t H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

/* ── SHA-256 compression ──────────────────────────────────────────────────── */

#ifdef USE_ARM_SHA2
/*
 * ARM hardware SHA-256 using vsha256h/vsha256h2/vsha256su0/vsha256su1.
 * Processes 4 rounds per pair of instructions using dedicated silicon.
 * ~4-8x faster than the generic C path on Qualcomm/Apple/Samsung ARM64 SoCs.
 */

/* 4 rounds with schedule expansion (rounds 0-47) */
#define R4SU(C, NXT, B, SC, OFF)                         \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));        \
    TMP2 = ABCD;                                          \
    C    = vsha256su0q_u32(C, NXT);                       \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);             \
    EFGH = vsha256h2q_u32(EFGH, TMP2, TMP0);             \
    C    = vsha256su1q_u32(C, B, SC)

/* 4 rounds, no schedule expansion (rounds 48-63) */
#define R4(C, OFF)                                        \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));        \
    TMP2 = ABCD;                                          \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);             \
    EFGH = vsha256h2q_u32(EFGH, TMP2, TMP0)

static void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32x4_t ABCD, EFGH, ABCD_SAVE, EFGH_SAVE;
    uint32x4_t MSG0, MSG1, MSG2, MSG3, TMP0, TMP2;

    ABCD = vld1q_u32(st);
    EFGH = vld1q_u32(st + 4);
    ABCD_SAVE = ABCD;
    EFGH_SAVE = EFGH;

    /* Load 64-byte block; vrev32q_u8 converts big-endian bytes to ARM uint32 */
    MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk)));
    MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 16)));
    MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 32)));
    MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 48)));

    /* Rounds 0-15 */
    R4SU(MSG0, MSG1, MSG2, MSG3,  0);
    R4SU(MSG1, MSG2, MSG3, MSG0,  4);
    R4SU(MSG2, MSG3, MSG0, MSG1,  8);
    R4SU(MSG3, MSG0, MSG1, MSG2, 12);
    /* Rounds 16-31 */
    R4SU(MSG0, MSG1, MSG2, MSG3, 16);
    R4SU(MSG1, MSG2, MSG3, MSG0, 20);
    R4SU(MSG2, MSG3, MSG0, MSG1, 24);
    R4SU(MSG3, MSG0, MSG1, MSG2, 28);
    /* Rounds 32-47 */
    R4SU(MSG0, MSG1, MSG2, MSG3, 32);
    R4SU(MSG1, MSG2, MSG3, MSG0, 36);
    R4SU(MSG2, MSG3, MSG0, MSG1, 40);
    R4SU(MSG3, MSG0, MSG1, MSG2, 44);
    /* Rounds 48-63: schedule fully expanded, just round computation */
    R4(MSG0, 48);
    R4(MSG1, 52);
    R4(MSG2, 56);
    R4(MSG3, 60);

    vst1q_u32(st,     vaddq_u32(ABCD, ABCD_SAVE));
    vst1q_u32(st + 4, vaddq_u32(EFGH, EFGH_SAVE));
}

#undef R4SU
#undef R4

#else  /* Generic C fallback (no ARM crypto extensions) */

#define RR(v,n) (((v)>>(n))|((v)<<(32-(n))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define EP1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define SG0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define SG1(x) (RR(x,17)^RR(x,19)^((x)>>10))

static void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t w[64], a, b, c, d, e, f, g, h, t1, t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)blk[i*4]   << 24) | ((uint32_t)blk[i*4+1] << 16)
             | ((uint32_t)blk[i*4+2] <<  8) |  (uint32_t)blk[i*4+3];
    for (i = 16; i < 64; i++)
        w[i] = SG1(w[i-2]) + w[i-7] + SG0(w[i-15]) + w[i-16];
    a=st[0]; b=st[1]; c=st[2]; d=st[3];
    e=st[4]; f=st[5]; g=st[6]; h=st[7];
    for (i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e,f,g) + K256[i] + w[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

#undef RR
#undef CH
#undef MAJ
#undef EP0
#undef EP1
#undef SG0
#undef SG1

#endif  /* USE_ARM_SHA2 */

/* ── Helpers ──────────────────────────────────────────────────────────────── */

/* Write 8 SHA-256 state words to 32 big-endian bytes */
static inline void state_to_bytes(const uint32_t st[8], uint8_t out[32]) {
    int i;
    for (i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

/* Returns 1 if hash < target, both treated as little-endian 256-bit integers.
 * Bitcoin pools (including public-pool.io) use le256todouble() to convert the
 * natural SHA-256d output bytes to a difficulty value, which means they check
 * the hash as a LE number: byte 31 is the most-significant byte.
 * A valid share has trailing zeros in natural byte order (= leading zeros when
 * displayed reversed, as shown on block explorers). */
static inline int hash_lt(const uint8_t hash[32], const uint8_t tgt[32]) {
    int i;
    for (i = 31; i >= 0; i--) {
        if (hash[i] < tgt[i]) return 1;
        if (hash[i] > tgt[i]) return 0;
    }
    return 0;
}

/* ── mine_range() ─────────────────────────────────────────────────────────── */

static PyObject* mine_range(PyObject* self, PyObject* args) {
    Py_buffer fb, sp, tb, stb;
    unsigned long long nonce_start, nonce_end;

    if (!PyArg_ParseTuple(args, "y*y*y*y*KK",
                          &fb, &sp, &tb, &stb, &nonce_start, &nonce_end))
        return NULL;

    PyObject* result = NULL;

    if (fb.len != 64 || sp.len != 12 || tb.len != 32 || stb.len != 32) {
        PyErr_SetString(PyExc_ValueError, "buffer sizes must be 64, 12, 32, 32");
        goto cleanup;
    }

    {
        const uint8_t* first_block    = (const uint8_t*)fb.buf;
        const uint8_t* second_prefix  = (const uint8_t*)sp.buf;
        const uint8_t* target_le      = (const uint8_t*)tb.buf;
        const uint8_t* soft_target_le = (const uint8_t*)stb.buf;

        /* Compute midstate from first 64-byte header block (once per call) */
        uint32_t midstate[8];
        memcpy(midstate, H0, sizeof(H0));
        sha256_compress(midstate, first_block);

        /*
         * Pre-build padded second block (64 bytes).
         * 80-byte message: [first_block:64][second_prefix:12][nonce:4]
         * SHA-256 padding for 80-byte input:
         *   [second_prefix:12][nonce:4][0x80][zeros:39][bitlen 640=0x280 big-endian:8]
         */
        uint8_t second_pad[64];
        memset(second_pad, 0, sizeof(second_pad));
        memcpy(second_pad, second_prefix, 12);
        second_pad[16] = 0x80u;
        second_pad[62] = 0x02u;
        second_pad[63] = 0x80u;

        /*
         * Pre-build outer padded block template (64 bytes).
         * 32-byte inner hash → SHA-256 padding:
         *   [inner_hash:32][0x80][zeros:23][bitlen 256=0x100 big-endian:8]
         */
        uint8_t outer_pad[64];
        memset(outer_pad, 0, sizeof(outer_pad));
        outer_pad[32] = 0x80u;
        outer_pad[62] = 0x01u;

        uint64_t hashes_done = 0;
        uint64_t soft_shares = 0;
        long long winner     = -1;

        /*
         * CRITICAL: release the GIL before the hot loop.
         * Without this, all 8 threads serialize on the GIL and only one
         * actually hashes at a time — 8 threads behaves like 1 thread.
         * After releasing, all C code runs truly in parallel on all cores.
         * All data accessed inside is in C buffers — no Python objects touched.
         */
        Py_BEGIN_ALLOW_THREADS

        {
            uint32_t inner_st[8], outer_st[8];
            uint8_t  hash_out[32];
            uint64_t nonce;

            for (nonce = nonce_start; nonce < nonce_end; nonce++) {
                /* Pack nonce little-endian into second block */
                second_pad[12] = (uint8_t)(nonce);
                second_pad[13] = (uint8_t)(nonce >>  8);
                second_pad[14] = (uint8_t)(nonce >> 16);
                second_pad[15] = (uint8_t)(nonce >> 24);

                /* Inner SHA-256: midstate + padded second block */
                memcpy(inner_st, midstate, 32);
                sha256_compress(inner_st, second_pad);
                state_to_bytes(inner_st, outer_pad);  /* inner hash → outer input */

                /* Outer SHA-256: fresh H0 + padded inner hash */
                memcpy(outer_st, H0, 32);
                sha256_compress(outer_st, outer_pad);
                state_to_bytes(outer_st, hash_out);

                hashes_done++;

                if (hash_lt(hash_out, target_le)) {
                    winner = (long long)nonce;
                    break;
                }
                if (hash_lt(hash_out, soft_target_le)) {
                    soft_shares++;
                }
            }
        }

        Py_END_ALLOW_THREADS

        if (winner >= 0)
            result = Py_BuildValue("(LKK)", winner, hashes_done, soft_shares);
        else
            result = Py_BuildValue("(OKK)", Py_None, hashes_done, soft_shares);
    }

cleanup:
    PyBuffer_Release(&fb);
    PyBuffer_Release(&sp);
    PyBuffer_Release(&tb);
    PyBuffer_Release(&stb);
    return result;
}

static PyMethodDef methods[] = {
    {"mine_range", mine_range, METH_VARARGS,
#ifdef USE_ARM_SHA2
     "SHA-256d mining loop — ARM hardware SHA-256 intrinsics + GIL released (v3)"
#else
     "SHA-256d mining loop — generic C SHA-256 + GIL released (v3)"
#endif
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "miner_core",
#ifdef USE_ARM_SHA2
    "Bitcoin SHA-256d inner loop v3: ARM SHA-256 hardware + true 8-thread parallelism.",
#else
    "Bitcoin SHA-256d inner loop v3: generic C SHA-256 + true 8-thread parallelism.",
#endif
    -1, methods
};

PyMODINIT_FUNC PyInit_miner_core(void) {
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
#ifdef USE_ARM_SHA2
    /* Runtime guard: compiled for ARM SHA2, but verify the CPU actually has it.
     * On rare devices without SHA2, return a clear error instead of a crash. */
    if (!(getauxval(AT_HWCAP) & HWCAP_SHA2)) {
        PyErr_SetString(PyExc_RuntimeError,
            "miner_core compiled with ARM SHA2 but this CPU does not support it. "
            "Rebuild with: python3 setup.py build_ext --inplace");
        Py_DECREF(m);
        return NULL;
    }
    PyModule_AddStringConstant(m, "path", "ARM_SHA2_hardware");
#else
    PyModule_AddStringConstant(m, "path", "generic_C");
#endif
    return m;
}
