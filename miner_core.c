/*
 * miner_core.c v4 — 4-way batched SHA-256d mining
 *
 * v4 vs v3:
 *   - Inner loop processes 4 nonces per iteration instead of 1.
 *   - 4 independent sha256_compress calls back-to-back expose enough
 *     independent work for the ARM64 OoO engine to hide the 3-cycle
 *     SHA2 instruction latency and drive throughput to ~1 SHA2 op/cycle.
 *   - second_pad copies are built once per mine_range call; only the
 *     4 nonce bytes are patched inside the hot loop (no memcpy inside loop).
 *   - outer_pad padding half (bytes 32-63) is also pre-built once.
 *   - Expected improvement: 2–3× over v3 on Cortex-A76/A78/X1/X2 cores.
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
 * 4 rounds per instruction pair; latency 3 cycles, throughput 1/cycle.
 * With 4 independent streams in the v4 hot loop, the OoO engine hides
 * the latency and drives all 4 SHA2 units at full throughput.
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

__attribute__((always_inline))
static inline void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
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

__attribute__((always_inline))
static inline void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
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

/* Patch a 32-bit nonce (little-endian) into bytes 12-15 of a block */
#define PATCH_NONCE(buf, n) do {             \
    (buf)[12] = (uint8_t)((n));              \
    (buf)[13] = (uint8_t)((n) >>  8);       \
    (buf)[14] = (uint8_t)((n) >> 16);       \
    (buf)[15] = (uint8_t)((n) >> 24);       \
} while (0)

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
         * Build second_pad template (64 bytes) from second_prefix + padding.
         * Layout: [second_prefix:12][nonce:4][0x80][zeros:38][0x0280 BE:2]
         * Only bytes 12-15 (nonce) vary per hash; everything else is fixed.
         */
        uint8_t second_pad[64];
        memset(second_pad, 0, sizeof(second_pad));
        memcpy(second_pad, second_prefix, 12);
        second_pad[16] = 0x80u;
        second_pad[62] = 0x02u;
        second_pad[63] = 0x80u;  /* bit length 640 = 0x0280 big-endian */

        uint64_t hashes_done = 0;
        uint64_t soft_shares = 0;
        long long winner     = -1;

        /*
         * Release the GIL: all 8 worker threads hash in parallel on real cores.
         * All data below is in C buffers; no Python objects are touched.
         */
        Py_BEGIN_ALLOW_THREADS

        {
            /*
             * 4-way batched inner loop.
             *
             * Four independent second_pad copies let the CPU see 4 unrelated
             * sha256_compress call chains simultaneously. On ARM64 with SHA2
             * extensions: SHA2 instructions have 3-cycle latency but 1-cycle
             * throughput — 4 independent streams fully hide the latency and
             * keep the SHA2 pipeline saturated.
             *
             * Bytes 0-11 and 16-63 of sp0..sp3 are identical (set once here).
             * Only bytes 12-15 (nonce) are patched inside the hot loop.
             */
            uint8_t sp0[64], sp1[64], sp2[64], sp3[64];
            memcpy(sp0, second_pad, 64);
            memcpy(sp1, second_pad, 64);
            memcpy(sp2, second_pad, 64);
            memcpy(sp3, second_pad, 64);

            /*
             * Four outer_pad buffers: bytes 32-63 are the SHA-256 padding for
             * a 32-byte (256-bit) inner hash — fixed for all nonces.
             * Bytes 0-31 are overwritten by state_to_bytes() each iteration.
             */
            uint8_t op0[64], op1[64], op2[64], op3[64];
            memset(op0, 0, 64); op0[32] = 0x80u; op0[62] = 0x01u;
            memset(op1, 0, 64); op1[32] = 0x80u; op1[62] = 0x01u;
            memset(op2, 0, 64); op2[32] = 0x80u; op2[62] = 0x01u;
            memset(op3, 0, 64); op3[32] = 0x80u; op3[62] = 0x01u;

            uint32_t is0[8], is1[8], is2[8], is3[8]; /* inner states */
            uint32_t os0[8], os1[8], os2[8], os3[8]; /* outer states */
            uint8_t  h0[32], h1[32], h2[32], h3[32]; /* final hashes */
            uint64_t nonce;

            /* ── 4-wide main loop ─────────────────────────────────────────── */
            for (nonce = nonce_start; nonce + 3 < nonce_end; nonce += 4) {
                uint32_t n0 = (uint32_t)nonce;
                uint32_t n1 = n0 + 1u;
                uint32_t n2 = n0 + 2u;
                uint32_t n3 = n0 + 3u;

                /* Patch nonce bytes only (16 byte-stores, no memcpy) */
                PATCH_NONCE(sp0, n0);
                PATCH_NONCE(sp1, n1);
                PATCH_NONCE(sp2, n2);
                PATCH_NONCE(sp3, n3);

                /*
                 * 4 independent inner SHA-256 compressions.
                 * The CPU's OoO engine sees no data dependency between the
                 * 4 chains and issues them into the SHA2 pipeline concurrently.
                 */
                memcpy(is0, midstate, 32); sha256_compress(is0, sp0);
                memcpy(is1, midstate, 32); sha256_compress(is1, sp1);
                memcpy(is2, midstate, 32); sha256_compress(is2, sp2);
                memcpy(is3, midstate, 32); sha256_compress(is3, sp3);

                /* Transfer inner hashes to outer_pad input region (bytes 0-31) */
                state_to_bytes(is0, op0);
                state_to_bytes(is1, op1);
                state_to_bytes(is2, op2);
                state_to_bytes(is3, op3);

                /* 4 independent outer SHA-256 compressions */
                memcpy(os0, H0, 32); sha256_compress(os0, op0);
                memcpy(os1, H0, 32); sha256_compress(os1, op1);
                memcpy(os2, H0, 32); sha256_compress(os2, op2);
                memcpy(os3, H0, 32); sha256_compress(os3, op3);

                /* Extract final SHA-256d hashes */
                state_to_bytes(os0, h0);
                state_to_bytes(os1, h1);
                state_to_bytes(os2, h2);
                state_to_bytes(os3, h3);

                hashes_done += 4;

                /* Check hard target (pool difficulty) */
                if (hash_lt(h0, target_le)) { winner = (long long)n0; break; }
                if (hash_lt(h1, target_le)) { winner = (long long)n1; break; }
                if (hash_lt(h2, target_le)) { winner = (long long)n2; break; }
                if (hash_lt(h3, target_le)) { winner = (long long)n3; break; }

                /* Count soft shares (hashrate calibration) */
                if (hash_lt(h0, soft_target_le)) soft_shares++;
                if (hash_lt(h1, soft_target_le)) soft_shares++;
                if (hash_lt(h2, soft_target_le)) soft_shares++;
                if (hash_lt(h3, soft_target_le)) soft_shares++;
            }

            /* ── 1-wide tail: handles remaining 0-3 nonces ───────────────── */
            if (winner < 0) {
                uint8_t sp_tail[64], op_tail[64], hash_out[32];
                uint32_t inner_st[8], outer_st[8];

                memcpy(sp_tail, second_pad, 64);
                memset(op_tail, 0, 64);
                op_tail[32] = 0x80u;
                op_tail[62] = 0x01u;

                for (; nonce < nonce_end; nonce++) {
                    PATCH_NONCE(sp_tail, (uint32_t)nonce);

                    memcpy(inner_st, midstate, 32);
                    sha256_compress(inner_st, sp_tail);
                    state_to_bytes(inner_st, op_tail);

                    memcpy(outer_st, H0, 32);
                    sha256_compress(outer_st, op_tail);
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

#undef PATCH_NONCE

static PyMethodDef methods[] = {
    {"mine_range", mine_range, METH_VARARGS,
#ifdef USE_ARM_SHA2
     "SHA-256d mining loop — ARM SHA2 hardware, 4-way batched OoO pipeline, GIL released (v4)"
#else
     "SHA-256d mining loop — generic C SHA-256, 4-way batched, GIL released (v4)"
#endif
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "miner_core",
#ifdef USE_ARM_SHA2
    "Bitcoin SHA-256d inner loop v4: ARM SHA2 + 4-way OoO batching + 8-thread parallelism.",
#else
    "Bitcoin SHA-256d inner loop v4: generic C SHA-256 + 4-way batching + 8-thread parallelism.",
#endif
    -1, methods
};

PyMODINIT_FUNC PyInit_miner_core(void) {
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
#ifdef USE_ARM_SHA2
    if (!(getauxval(AT_HWCAP) & HWCAP_SHA2)) {
        PyErr_SetString(PyExc_RuntimeError,
            "miner_core compiled with ARM SHA2 but this CPU does not support it. "
            "Rebuild with: python3 setup.py build_ext --inplace");
        Py_DECREF(m);
        return NULL;
    }
    PyModule_AddStringConstant(m, "path", "ARM_SHA2_4way");
#else
    PyModule_AddStringConstant(m, "path", "generic_C_4way");
#endif
    return m;
}
