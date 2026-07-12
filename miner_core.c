/*
 * miner_core.c v2 — Zero-overhead Bitcoin SHA-256d mining inner loop
 *
 * v2 changes vs v1:
 *   - Custom sha256_compress() replaces all OpenSSL calls in the hot loop
 *   - Pre-built padded blocks eliminate per-nonce SHA256_Final padding work
 *   - 32-byte midstate copy replaces 112-byte SHA256_CTX memcpy
 *   - No OpenSSL dependency at all (pure C, no -lcrypto needed)
 *
 * Expected speedup over v1: 2-4x on ARM64 with -O3 -march=native
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

/* ── Compression function ─────────────────────────────────────────────────── */

#define RR(v,n) (((v)>>(n))|((v)<<(32-(n))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define EP1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define SG0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define SG1(x) (RR(x,17)^RR(x,19)^((x)>>10))

/*
 * Process one 64-byte block through SHA-256 compression.
 * st[8] is updated in place (additive). blk[64] is big-endian byte order.
 * With -O3 -march=native, this inlines and the compiler unrolls the 64 rounds.
 */
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

/* Returns 1 if hash < target (both interpreted as 32-byte little-endian integers) */
static inline int hash_lt(const uint8_t hash[32], const uint8_t tgt[32]) {
    int i;
    for (i = 31; i >= 0; i--) {
        if (hash[i] < tgt[i]) return 1;
        if (hash[i] > tgt[i]) return 0;
    }
    return 0;
}

/* ── Python-visible mine_range() ─────────────────────────────────────────── */

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

        /* ── Compute SHA-256 midstate (once per call) ────────────────────── */
        uint32_t midstate[8];
        memcpy(midstate, H0, sizeof(H0));
        sha256_compress(midstate, first_block);

        /*
         * Pre-build padded second block (64 bytes).
         * 80-byte message: first_block(64) + second_prefix(12) + nonce(4)
         * SHA-256 padding for 80-byte input:
         *   [second_prefix:12][nonce:4][0x80][zeros:39][bitlen:8]
         *   bit length = 80*8 = 640 = 0x0000000000000280
         *   → bytes [62-63] = 0x02, 0x80  (rest of bitlen is zero)
         */
        uint8_t second_pad[64];
        memset(second_pad, 0, sizeof(second_pad));
        memcpy(second_pad, second_prefix, 12);
        /* [12-15]: nonce (written per iteration) */
        second_pad[16] = 0x80u;
        second_pad[62] = 0x02u;
        second_pad[63] = 0x80u;

        /*
         * Pre-build outer padded block template (64 bytes).
         * 32-byte inner hash → SHA-256 padding:
         *   [inner_hash:32][0x80][zeros:23][bitlen:8]
         *   bit length = 32*8 = 256 = 0x0000000000000100
         *   → bytes [62-63] = 0x01, 0x00
         */
        uint8_t outer_pad[64];
        memset(outer_pad, 0, sizeof(outer_pad));
        /* [0-31]: inner hash (written per iteration) */
        outer_pad[32] = 0x80u;
        outer_pad[62] = 0x01u;
        /* outer_pad[63] = 0x00 already */

        /* ── Hot loop ─────────────────────────────────────────────────────── */
        uint64_t hashes_done = 0;
        uint64_t soft_shares = 0;
        long long winner     = -1;
        uint32_t inner_st[8], outer_st[8];
        uint8_t  hash_out[32];

        for (uint64_t nonce = nonce_start; nonce < nonce_end; nonce++) {
            /* Pack nonce little-endian into second block */
            second_pad[12] = (uint8_t)(nonce);
            second_pad[13] = (uint8_t)(nonce >>  8);
            second_pad[14] = (uint8_t)(nonce >> 16);
            second_pad[15] = (uint8_t)(nonce >> 24);

            /* Inner SHA-256: midstate + padded 64-byte second block */
            memcpy(inner_st, midstate, 32);
            sha256_compress(inner_st, second_pad);

            /* Inner hash → outer_pad[0-31] */
            state_to_bytes(inner_st, outer_pad);

            /* Outer SHA-256: fresh H0 + padded 64-byte inner hash */
            memcpy(outer_st, H0, 32);
            sha256_compress(outer_st, outer_pad);

            /* Extract final 32-byte hash */
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
     "mine_range(first_block, second_prefix, target_le, soft_target_le,\n"
     "           nonce_start, nonce_end) -> (winner|None, hashes_done, soft_shares)\n\n"
     "SHA-256d Bitcoin inner loop — custom compression fn, no OpenSSL, pre-padded blocks."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "miner_core",
    "Zero-overhead SHA-256d mining loop (v2: pure C, no OpenSSL).",
    -1, methods
};

PyMODINIT_FUNC PyInit_miner_core(void) {
    return PyModule_Create(&module);
}
