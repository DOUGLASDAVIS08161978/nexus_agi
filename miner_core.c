/*
 * miner_core.c v6 — zero-copy SHA-256d: no memcpy, no state_to_bytes in hot loop
 *
 * v6 vs v5:
 *   sha256_inner2(sa, ba, sb, bb, midstate)
 *     Takes the shared initial state (midstate) directly — no memcpy(is, midstate)
 *     needed before the call. Both streams start from the same midstate value
 *     loaded once.
 *
 *   sha256_outer2(sa, inner_a, sb, inner_b)
 *     Takes the uint32 inner-hash state arrays directly — no state_to_bytes(),
 *     no outer_pad byte buffers, no vrev32q_u8 on input. The inner state words
 *     written by vst1q_u32 are the exact values SHA2 instructions need when
 *     loaded back with vld1q_u32 — the byte-swap round-trip was a no-op.
 *     H0 initial state is hardcoded; padding words (0x80000000, 0x100) are
 *     compile-time constants.
 *
 * Net savings per 4-nonce hot-loop iteration:
 *   - 4 × memcpy(is, midstate, 32)  eliminated
 *   - 4 × memcpy(os, H0, 32)        eliminated
 *   - 4 × state_to_bytes(32 stores) eliminated
 *   - 4 × outer_pad byte buffers    eliminated (256 bytes of stack)
 *   - 8 × vrev32q_u8 in outer load  eliminated
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

/* ── single-stream compress (midstate init + tail loop) ────────────────── */
#define R4SU(C, NXT, B, SC, OFF)                          \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));         \
    TMP2 = ABCD;                                           \
    C    = vsha256su0q_u32(C, NXT);                        \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);              \
    EFGH = vsha256h2q_u32(EFGH, TMP2, TMP0);              \
    C    = vsha256su1q_u32(C, B, SC)

#define R4(C, OFF)                                         \
    TMP0 = vaddq_u32(C, vld1q_u32(K256 + (OFF)));         \
    TMP2 = ABCD;                                           \
    ABCD = vsha256hq_u32 (ABCD, EFGH, TMP0);              \
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

/* ── 2-way interleaved round macros (shared by inner2 and outer2) ──────── */
/*
 * RA4/RB4: 4 rounds + schedule expansion for stream A / stream B.
 * Caller interleaves RA4 immediately followed by RB4 so that B's 6
 * independent instructions run during A's 3-cycle vsha256hq latency stall.
 * Token-paste: RA4(M0,M1,M2,M3,OFF) → uses M0a, M1a … Aa, Ea, T0, T2.
 */
#define RA4(C,N,B,S,OFF)                                                      \
    T0=vaddq_u32(C##a,vld1q_u32(K256+(OFF))); T2=Aa;                         \
    C##a=vsha256su0q_u32(C##a,N##a);                                          \
    Aa=vsha256hq_u32(Aa,Ea,T0); Ea=vsha256h2q_u32(Ea,T2,T0);                \
    C##a=vsha256su1q_u32(C##a,B##a,S##a)

#define RB4(C,N,B,S,OFF)                                                      \
    T0=vaddq_u32(C##b,vld1q_u32(K256+(OFF))); T2=Ab;                         \
    C##b=vsha256su0q_u32(C##b,N##b);                                          \
    Ab=vsha256hq_u32(Ab,Eb,T0); Eb=vsha256h2q_u32(Eb,T2,T0);                \
    C##b=vsha256su1q_u32(C##b,B##b,S##b)

#define RA4F(C,OFF)                                                           \
    T0=vaddq_u32(C##a,vld1q_u32(K256+(OFF))); T2=Aa;                         \
    Aa=vsha256hq_u32(Aa,Ea,T0); Ea=vsha256h2q_u32(Ea,T2,T0)

#define RB4F(C,OFF)                                                           \
    T0=vaddq_u32(C##b,vld1q_u32(K256+(OFF))); T2=Ab;                         \
    Ab=vsha256hq_u32(Ab,Eb,T0); Eb=vsha256h2q_u32(Eb,T2,T0)

/*
 * sha256_inner2 — 2-way interleaved inner-block compression.
 *
 * Takes the shared midstate directly (init) — both streams start from the
 * same initial state. Eliminates 2× memcpy(is, midstate, 32) per call.
 * Writes results to sa[8] and sb[8].
 */
static void sha256_inner2(
        uint32_t sa[8], const uint8_t ba[64],
        uint32_t sb[8], const uint8_t bb[64],
        const uint32_t init[8])
{
    uint32x4_t Aa, Ea, SaveAa, SaveEa;
    uint32x4_t Ab, Eb, SaveAb, SaveEb;
    uint32x4_t M0a, M1a, M2a, M3a;
    uint32x4_t M0b, M1b, M2b, M3b;
    uint32x4_t T0, T2;

    /* Both streams share the midstate as initial state */
    Aa = vld1q_u32(init);    Ea = vld1q_u32(init + 4);
    Ab = Aa;                  Eb = Ea;
    SaveAa = Aa; SaveEa = Ea;
    SaveAb = Aa; SaveEb = Ea;

    /* Load both blocks (differ only at nonce bytes 12-15) */
    M0a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba)));
    M1a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+16)));
    M2a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+32)));
    M3a=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(ba+48)));

    M0b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb)));
    M1b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+16)));
    M2b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+32)));
    M3b=vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(bb+48)));

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
    RA4F(M0,48); RB4F(M0,48);
    RA4F(M1,52); RB4F(M1,52);
    RA4F(M2,56); RB4F(M2,56);
    RA4F(M3,60); RB4F(M3,60);

    vst1q_u32(sa,   vaddq_u32(Aa, SaveAa));
    vst1q_u32(sa+4, vaddq_u32(Ea, SaveEa));
    vst1q_u32(sb,   vaddq_u32(Ab, SaveAb));
    vst1q_u32(sb+4, vaddq_u32(Eb, SaveEb));
}

/*
 * sha256_outer2 — 2-way interleaved outer-block compression.
 *
 * Takes uint32 inner-hash state arrays directly — no state_to_bytes(),
 * no byte buffers, no vrev32q_u8 on input. The inner state words stored by
 * vst1q_u32 are identical to what vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8))
 * would recover after a state_to_bytes round-trip, so we skip both steps.
 *
 * H0 initial state is hardcoded as compile-time constants (no load from
 * the H0 array, no memcpy). Padding words are also hardcoded:
 *   W[8]  = 0x80000000  (padding bit for 256-bit message)
 *   W[9..14] = 0
 *   W[15] = 0x00000100  (256 bits in big-endian 64-bit length field)
 */
static void sha256_outer2(
        uint32_t sa[8], const uint32_t inner_a[8],
        uint32_t sb[8], const uint32_t inner_b[8])
{
    uint32x4_t Aa, Ea, SaveAa, SaveEa;
    uint32x4_t Ab, Eb, SaveAb, SaveEb;
    uint32x4_t M0a, M1a, M2a, M3a;
    uint32x4_t M0b, M1b, M2b, M3b;
    uint32x4_t T0, T2;

    /* Hardcoded H0 — compiler generates immediate vector loads, no memory reads */
    Aa = vcombine_u32(vcreate_u32(0xbb67ae856a09e667ULL),
                      vcreate_u32(0xa54ff53a3c6ef372ULL));
    Ea = vcombine_u32(vcreate_u32(0x9b05688c510e527fULL),
                      vcreate_u32(0x5be0cd191f83d9abULL));
    Ab = Aa; Eb = Ea;
    SaveAa = Aa; SaveEa = Ea;
    SaveAb = Aa; SaveEb = Ea;

    /*
     * Load inner hash words as uint32 — no byte reversal needed.
     * The inner state written by vst1q_u32 is already in the format
     * that SHA2 instructions consume when loaded with vld1q_u32.
     */
    M0a = vld1q_u32(inner_a);      /* W[0..3]: inner hash words 0-3 */
    M1a = vld1q_u32(inner_a + 4);  /* W[4..7]: inner hash words 4-7 */
    M0b = vld1q_u32(inner_b);
    M1b = vld1q_u32(inner_b + 4);

    /* Hardcoded padding: W[8]=0x80000000, W[9..14]=0, W[15]=0x00000100 */
    M2a = M2b = vcombine_u32(vcreate_u32(0x0000000080000000ULL),
                              vcreate_u32(0x0000000000000000ULL));
    M3a = M3b = vcombine_u32(vcreate_u32(0x0000000000000000ULL),
                              vcreate_u32(0x0000010000000000ULL));

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

#else  /* Generic C fallback */

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

static void sha256_inner2(
        uint32_t sa[8], const uint8_t ba[64],
        uint32_t sb[8], const uint8_t bb[64],
        const uint32_t init[8]) {
    memcpy(sa, init, 32); sha256_compress(sa, ba);
    memcpy(sb, init, 32); sha256_compress(sb, bb);
}

static void sha256_outer2(
        uint32_t sa[8], const uint32_t inner_a[8],
        uint32_t sb[8], const uint32_t inner_b[8]) {
    uint8_t op[64]; int i;
    memset(op, 0, 64); op[32]=0x80u; op[62]=0x01u;

    memcpy(sa, H0, 32);
    for (i = 0; i < 8; i++) {
        op[i*4]=(uint8_t)(inner_a[i]>>24); op[i*4+1]=(uint8_t)(inner_a[i]>>16);
        op[i*4+2]=(uint8_t)(inner_a[i]>>8); op[i*4+3]=(uint8_t)(inner_a[i]);
    }
    sha256_compress(sa, op);

    memcpy(sb, H0, 32);
    for (i = 0; i < 8; i++) {
        op[i*4]=(uint8_t)(inner_b[i]>>24); op[i*4+1]=(uint8_t)(inner_b[i]>>16);
        op[i*4+2]=(uint8_t)(inner_b[i]>>8); op[i*4+3]=(uint8_t)(inner_b[i]);
    }
    sha256_compress(sb, op);
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

        /* Midstate: compress first 64-byte header block once per call */
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
             * Hot-loop layout (4 nonces per iteration):
             *
             *   sha256_inner2(is0,sp0, is1,sp1, midstate)  — inner pair 0+1
             *   sha256_inner2(is2,sp2, is3,sp3, midstate)  — inner pair 2+3
             *   sha256_outer2(os0,is0, os1,is1)             — outer pair 0+1
             *   sha256_outer2(os2,is2, os3,is3)             — outer pair 2+3
             *   state_to_bytes × 4 → h0..h3                 — extract hashes
             *
             * No memcpy, no state_to_bytes for intermediate transfer,
             * no outer_pad byte buffers. The outer_pad savings alone
             * eliminate 128 stores + 8 vrev32q_u8 per 4-nonce iteration.
             */

            /* Second-pad copies: only nonce bytes (12-15) patched per iter */
            uint8_t sp0[64], sp1[64], sp2[64], sp3[64];
            memcpy(sp0, second_pad, 64);
            memcpy(sp1, second_pad, 64);
            memcpy(sp2, second_pad, 64);
            memcpy(sp3, second_pad, 64);

            uint32_t is0[8], is1[8], is2[8], is3[8];  /* inner states */
            uint32_t os0[8], os1[8], os2[8], os3[8];  /* outer states */
            uint8_t  h0[32], h1[32], h2[32], h3[32];  /* final hashes */
            uint64_t nonce;

            /* ── 4-wide main loop ─────────────────────────────────────────── */
            for (nonce = nonce_start; nonce + 3 < nonce_end; nonce += 4) {
                uint32_t n0 = (uint32_t)nonce;
                uint32_t n1 = n0+1u, n2 = n0+2u, n3 = n0+3u;

                PATCH_NONCE(sp0, n0); PATCH_NONCE(sp1, n1);
                PATCH_NONCE(sp2, n2); PATCH_NONCE(sp3, n3);

                /* Inner: 2 interleaved pairs — no memcpy of midstate needed */
                sha256_inner2(is0, sp0, is1, sp1, midstate);
                sha256_inner2(is2, sp2, is3, sp3, midstate);

                /* Outer: 2 interleaved pairs — no state_to_bytes, no byte bufs */
                sha256_outer2(os0, is0, os1, is1);
                sha256_outer2(os2, is2, os3, is3);

                /* Extract final SHA-256d hash bytes for comparison */
                state_to_bytes(os0, h0);
                state_to_bytes(os1, h1);
                state_to_bytes(os2, h2);
                state_to_bytes(os3, h3);

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

            /* ── 1-wide tail: 0-3 remaining nonces ───────────────────────── */
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
     "SHA-256d: ARM SHA2 2-way interleaved, zero-copy hot loop, GIL released (v6)"
#else
     "SHA-256d: generic C, 4-way batched, GIL released (v6)"
#endif
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "miner_core",
#ifdef USE_ARM_SHA2
    "Bitcoin SHA-256d v6: ARM SHA2 2-way interleaved, zero-copy, 8-thread parallel.",
#else
    "Bitcoin SHA-256d v6: generic C + 8-thread parallel.",
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
    PyModule_AddStringConstant(m, "path", "ARM_SHA2_v6_zerocopy");
#else
    PyModule_AddStringConstant(m, "path", "generic_C_v6");
#endif
    return m;
}
