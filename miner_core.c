/*
 * miner_core.c v5 — 2-way interleaved + 4-way batched SHA-256d mining
 *
 * v5 vs v4:
 *   - Replaces 8× non-inlined sha256_compress() per 4-way iteration with
 *     4× sha256_compress2() calls — each processes 2 independent blocks
 *     with explicitly interleaved ARM SHA2 instructions.
 *   - v4+always_inline caused register spilling (80 virtual NEON regs vs
 *     32 physical) which hurt performance. sha256_compress2 uses exactly
 *     18 NEON registers — no spilling.
 *   - The explicit A/B interleaving hides the 3-cycle SHA2 instruction
 *     latency: while stream A's vsha256hq result is baking (3 cycles),
 *     stream B's independent instructions execute, keeping the SHA2
 *     hardware at ~1-cycle throughput instead of 1-per-3-cycle stall.
 *   - Expected improvement over v4: ~1.5–2× on big OoO cores (A76/A78/X).
 *     Little in-order cores (A55/A510) see same throughput as v4.
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

/* SHA-256 initial hash values */
static const uint32_t H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

/* ── SHA-256 compression ──────────────────────────────────────────────────── */

#ifdef USE_ARM_SHA2

/*
 * Single-stream ARM SHA-256 compression.
 * Used by the tail loop (0-3 leftover nonces per mine_range batch).
 */
#define R4SU(C, NXT, B, SC, OFF)                         \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));        \
    TMP2 = ABCD;                                          \
    C    = vsha256su0q_u32(C, NXT);                       \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);             \
    EFGH = vsha256h2q_u32(EFGH, TMP2, TMP0);             \
    C    = vsha256su1q_u32(C, B, SC)

#define R4(C, OFF)                                        \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));        \
    TMP2 = ABCD;                                          \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);             \
    EFGH = vsha256h2q_u32(EFGH, TMP2, TMP0)

static void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32x4_t ABCD, EFGH, ABCD_SAVE, EFGH_SAVE;
    uint32x4_t MSG0, MSG1, MSG2, MSG3, TMP0, TMP2;

    ABCD = vld1q_u32(st);   EFGH = vld1q_u32(st + 4);
    ABCD_SAVE = ABCD;       EFGH_SAVE = EFGH;

    MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk)));
    MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 16)));
    MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 32)));
    MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 48)));

    R4SU(MSG0, MSG1, MSG2, MSG3,  0);
    R4SU(MSG1, MSG2, MSG3, MSG0,  4);
    R4SU(MSG2, MSG3, MSG0, MSG1,  8);
    R4SU(MSG3, MSG0, MSG1, MSG2, 12);
    R4SU(MSG0, MSG1, MSG2, MSG3, 16);
    R4SU(MSG1, MSG2, MSG3, MSG0, 20);
    R4SU(MSG2, MSG3, MSG0, MSG1, 24);
    R4SU(MSG3, MSG0, MSG1, MSG2, 28);
    R4SU(MSG0, MSG1, MSG2, MSG3, 32);
    R4SU(MSG1, MSG2, MSG3, MSG0, 36);
    R4SU(MSG2, MSG3, MSG0, MSG1, 40);
    R4SU(MSG3, MSG0, MSG1, MSG2, 44);
    R4(MSG0, 48); R4(MSG1, 52); R4(MSG2, 56); R4(MSG3, 60);

    vst1q_u32(st,     vaddq_u32(ABCD, ABCD_SAVE));
    vst1q_u32(st + 4, vaddq_u32(EFGH, EFGH_SAVE));
}

#undef R4SU
#undef R4

/*
 * 2-way interleaved ARM SHA-256 compression (the hot-path workhorse).
 *
 * Processes 2 independent 64-byte blocks (streams A and B) simultaneously.
 * Operations for A and B are explicitly interleaved: while A's vsha256hq
 * result is pending (3-cycle hardware latency), B's independent loads,
 * adds, and schedule-expansion instructions fill those pipeline slots,
 * driving the SHA2 execution units at ~1-cycle throughput.
 *
 * Register budget: 18 NEON registers — well within ARM64's 32 (no spilling).
 *   Aa/Ea/SaveAa/SaveEa   — stream A state   (4)
 *   Ab/Eb/SaveAb/SaveEb   — stream B state   (4)
 *   M0a–M3a               — stream A schedule (4)
 *   M0b–M3b               — stream B schedule (4)
 *   T0/T2                 — shared temporaries (2)
 */

/*
 * Macro token-pasting trick: RA4(M0,M1,M2,M3,OFF) expands to use M0a, M1a etc.
 * RB4 does the same for M0b, M1b etc. Caller interleaves RA4 then RB4 per
 * group so that B's instructions run during A's 3-cycle latency stall.
 */
#define RA4(C,N,B,S,OFF)                                                     \
    T0=vaddq_u32(C##a,vld1q_u32(K256+(OFF))); T2=Aa;                        \
    C##a=vsha256su0q_u32(C##a,N##a);                                         \
    Aa=vsha256hq_u32(Aa,Ea,T0); Ea=vsha256h2q_u32(Ea,T2,T0);               \
    C##a=vsha256su1q_u32(C##a,B##a,S##a)

#define RB4(C,N,B,S,OFF)                                                     \
    T0=vaddq_u32(C##b,vld1q_u32(K256+(OFF))); T2=Ab;                        \
    C##b=vsha256su0q_u32(C##b,N##b);                                         \
    Ab=vsha256hq_u32(Ab,Eb,T0); Eb=vsha256h2q_u32(Eb,T2,T0);               \
    C##b=vsha256su1q_u32(C##b,B##b,S##b)

/* Rounds 48-63: schedule fully expanded, just round computation */
#define RA4F(C,OFF)                                                          \
    T0=vaddq_u32(C##a,vld1q_u32(K256+(OFF))); T2=Aa;                        \
    Aa=vsha256hq_u32(Aa,Ea,T0); Ea=vsha256h2q_u32(Ea,T2,T0)

#define RB4F(C,OFF)                                                          \
    T0=vaddq_u32(C##b,vld1q_u32(K256+(OFF))); T2=Ab;                        \
    Ab=vsha256hq_u32(Ab,Eb,T0); Eb=vsha256h2q_u32(Eb,T2,T0)

static void sha256_compress2(
        uint32_t sa[8], const uint8_t ba[64],
        uint32_t sb[8], const uint8_t bb[64])
{
    /* State: 4 regs each stream × 2 streams = 8 */
    uint32x4_t Aa, Ea, SaveAa, SaveEa;
    uint32x4_t Ab, Eb, SaveAb, SaveEb;
    /* Schedule: 4 regs each stream × 2 streams = 8 */
    uint32x4_t M0a, M1a, M2a, M3a;
    uint32x4_t M0b, M1b, M2b, M3b;
    /* Shared temporaries: 2 */
    uint32x4_t T0, T2;

    Aa=vld1q_u32(sa);   Ea=vld1q_u32(sa+4); SaveAa=Aa; SaveEa=Ea;
    Ab=vld1q_u32(sb);   Eb=vld1q_u32(sb+4); SaveAb=Ab; SaveEb=Eb;

    M0a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba)));
    M1a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+16)));
    M2a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+32)));
    M3a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+48)));

    M0b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb)));
    M1b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+16)));
    M2b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+32)));
    M3b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+48)));

    /*
     * Rounds 0–47: 12 groups of 4. Each group: A first, then B.
     * B's 6 instructions fill the 3-cycle stall from A's vsha256hq/h2q.
     */
    RA4(M0,M1,M2,M3, 0); RB4(M0,M1,M2,M3, 0);
    RA4(M1,M2,M3,M0, 4); RB4(M1,M2,M3,M0, 4);
    RA4(M2,M3,M0,M1, 8); RB4(M2,M3,M0,M1, 8);
    RA4(M3,M0,M1,M2,12); RB4(M3,M0,M1,M2,12);
    RA4(M0,M1,M2,M3,16); RB4(M0,M1,M2,M3,16);
    RA4(M1,M2,M3,M0,20); RB4(M1,M2,M3,M0,20);
    RA4(M2,M3,M0,M1,24); RB4(M2,M3,M0,M1,24);
    RA4(M3,M0,M1,M2,28); RB4(M3,M0,M1,M2,28);
    RA4(M0,M1,M2,M3,32); RB4(M0,M1,M2,M3,32);
    RA4(M1,M2,M3,M0,36); RB4(M1,M2,M3,M0,36);
    RA4(M2,M3,M0,M1,40); RB4(M2,M3,M0,M1,40);
    RA4(M3,M0,M1,M2,44); RB4(M3,M0,M1,M2,44);

    /* Rounds 48–63: no more schedule expansion */
    RA4F(M0,48); RB4F(M0,48);
    RA4F(M1,52); RB4F(M1,52);
    RA4F(M2,56); RB4F(M2,56);
    RA4F(M3,60); RB4F(M3,60);

    vst1q_u32(sa,   vaddq_u32(Aa, SaveAa));
    vst1q_u32(sa+4, vaddq_u32(Ea, SaveEa));
    vst1q_u32(sb,   vaddq_u32(Ab, SaveAb));
    vst1q_u32(sb+4, vaddq_u32(Eb, SaveEb));
}

#undef RA4
#undef RB4
#undef RA4F
#undef RB4F

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

/* Generic 2-way: just call compress twice (no hardware latency to hide) */
static void sha256_compress2(
        uint32_t sa[8], const uint8_t ba[64],
        uint32_t sb[8], const uint8_t bb[64]) {
    sha256_compress(sa, ba);
    sha256_compress(sb, bb);
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

static inline void state_to_bytes(const uint32_t st[8], uint8_t out[32]) {
    int i;
    for (i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

/* LE comparison: Bitcoin pools interpret hash bytes as a little-endian 256-bit
 * number (le256todouble). Byte 31 is the MSB; a valid share has trailing zeros
 * in natural byte order (= leading zeros on a block explorer). */
static inline int hash_lt(const uint8_t hash[32], const uint8_t tgt[32]) {
    int i;
    for (i = 31; i >= 0; i--) {
        if (hash[i] < tgt[i]) return 1;
        if (hash[i] > tgt[i]) return 0;
    }
    return 0;
}

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

        uint32_t midstate[8];
        memcpy(midstate, H0, sizeof(H0));
        sha256_compress(midstate, first_block);

        /* second_pad template: [second_prefix:12][nonce:4][SHA256 padding:48] */
        uint8_t second_pad[64];
        memset(second_pad, 0, sizeof(second_pad));
        memcpy(second_pad, second_prefix, 12);
        second_pad[16] = 0x80u;
        second_pad[62] = 0x02u;
        second_pad[63] = 0x80u;

        uint64_t hashes_done = 0;
        uint64_t soft_shares = 0;
        long long winner     = -1;

        Py_BEGIN_ALLOW_THREADS

        {
            /*
             * 4-way batched loop using sha256_compress2 for each pair.
             *
             * Inner iteration per 4 nonces:
             *   sha256_compress2(is0,sp0, is1,sp1)  ← inner hashes for n+0, n+1
             *   sha256_compress2(is2,sp2, is3,sp3)  ← inner hashes for n+2, n+3
             *   sha256_compress2(os0,op0, os1,op1)  ← outer hashes for n+0, n+1
             *   sha256_compress2(os2,op2, os3,op3)  ← outer hashes for n+2, n+3
             *
             * The OoO engine also sees independence between the two sha256_compress2
             * calls for inner (and for outer), giving additional overlap.
             */

            /* Second-pad copies: built once, only nonce bytes (12-15) patched */
            uint8_t sp0[64], sp1[64], sp2[64], sp3[64];
            memcpy(sp0, second_pad, 64);
            memcpy(sp1, second_pad, 64);
            memcpy(sp2, second_pad, 64);
            memcpy(sp3, second_pad, 64);

            /* Outer-pad buffers: bytes 32-63 are fixed SHA256 padding */
            uint8_t op0[64], op1[64], op2[64], op3[64];
            memset(op0, 0, 64); op0[32]=0x80u; op0[62]=0x01u;
            memset(op1, 0, 64); op1[32]=0x80u; op1[62]=0x01u;
            memset(op2, 0, 64); op2[32]=0x80u; op2[62]=0x01u;
            memset(op3, 0, 64); op3[32]=0x80u; op3[62]=0x01u;

            uint32_t is0[8], is1[8], is2[8], is3[8];
            uint32_t os0[8], os1[8], os2[8], os3[8];
            uint8_t  h0[32], h1[32], h2[32], h3[32];
            uint64_t nonce;

            /* ── 4-wide main loop ─────────────────────────────────────────── */
            for (nonce = nonce_start; nonce + 3 < nonce_end; nonce += 4) {
                uint32_t n0 = (uint32_t)nonce;
                uint32_t n1 = n0 + 1u, n2 = n0 + 2u, n3 = n0 + 3u;

                PATCH_NONCE(sp0, n0); PATCH_NONCE(sp1, n1);
                PATCH_NONCE(sp2, n2); PATCH_NONCE(sp3, n3);

                /* Inner SHA-256: pair (n0,n1) then pair (n2,n3) — interleaved */
                memcpy(is0, midstate, 32); memcpy(is1, midstate, 32);
                sha256_compress2(is0, sp0, is1, sp1);

                memcpy(is2, midstate, 32); memcpy(is3, midstate, 32);
                sha256_compress2(is2, sp2, is3, sp3);

                /* Move inner hashes into outer-pad input regions */
                state_to_bytes(is0, op0); state_to_bytes(is1, op1);
                state_to_bytes(is2, op2); state_to_bytes(is3, op3);

                /* Outer SHA-256: pair (n0,n1) then pair (n2,n3) — interleaved */
                memcpy(os0, H0, 32); memcpy(os1, H0, 32);
                sha256_compress2(os0, op0, os1, op1);

                memcpy(os2, H0, 32); memcpy(os3, H0, 32);
                sha256_compress2(os2, op2, os3, op3);

                state_to_bytes(os0, h0); state_to_bytes(os1, h1);
                state_to_bytes(os2, h2); state_to_bytes(os3, h3);

                hashes_done += 4;

                if (hash_lt(h0, target_le)) { winner=(long long)n0; break; }
                if (hash_lt(h1, target_le)) { winner=(long long)n1; break; }
                if (hash_lt(h2, target_le)) { winner=(long long)n2; break; }
                if (hash_lt(h3, target_le)) { winner=(long long)n3; break; }

                if (hash_lt(h0, soft_target_le)) soft_shares++;
                if (hash_lt(h1, soft_target_le)) soft_shares++;
                if (hash_lt(h2, soft_target_le)) soft_shares++;
                if (hash_lt(h3, soft_target_le)) soft_shares++;
            }

            /* ── 1-wide tail: handles remaining 0-3 nonces ───────────────── */
            if (winner < 0) {
                uint8_t sp_t[64], op_t[64], hash_out[32];
                uint32_t is[8], os[8];

                memcpy(sp_t, second_pad, 64);
                memset(op_t, 0, 64);
                op_t[32]=0x80u; op_t[62]=0x01u;

                for (; nonce < nonce_end; nonce++) {
                    PATCH_NONCE(sp_t, (uint32_t)nonce);

                    memcpy(is, midstate, 32);
                    sha256_compress(is, sp_t);
                    state_to_bytes(is, op_t);

                    memcpy(os, H0, 32);
                    sha256_compress(os, op_t);
                    state_to_bytes(os, hash_out);

                    hashes_done++;
                    if (hash_lt(hash_out, target_le)) { winner=(long long)nonce; break; }
                    if (hash_lt(hash_out, soft_target_le)) soft_shares++;
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
     "SHA-256d: ARM SHA2 2-way interleaved + 4-way batched, GIL released (v5)"
#else
     "SHA-256d: generic C + 4-way batched, GIL released (v5)"
#endif
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "miner_core",
#ifdef USE_ARM_SHA2
    "Bitcoin SHA-256d v5: ARM SHA2 2-way interleaved + 4-way batching + 8-thread parallel.",
#else
    "Bitcoin SHA-256d v5: generic C + 4-way batching + 8-thread parallel.",
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
            "Rebuild: python3 setup.py build_ext --inplace");
        Py_DECREF(m);
        return NULL;
    }
    PyModule_AddStringConstant(m, "path", "ARM_SHA2_2way_interleaved");
#else
    PyModule_AddStringConstant(m, "path", "generic_C_4way");
#endif
    return m;
}
