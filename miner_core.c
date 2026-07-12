/*
 * miner_core.c — Native Bitcoin SHA-256d mining inner loop for Nova Miner
 *
 * Build in Termux:
 *   pkg install clang python-dev
 *   python3 setup.py build_ext --inplace
 *
 * If it loads, the mining loop runs 100% in C — no Python interpreter overhead
 * between hashes. Expected speedup: 10-50x over pure Python hashlib path.
 *
 * API:
 *   mine_range(first_block, second_prefix, target_le, soft_target_le,
 *              nonce_start, nonce_end)
 *     -> (winning_nonce_or_None, hashes_done, soft_shares_found)
 *
 *   first_block    : bytes[64]  version + prevhash + merkle[:28]  (constant per job)
 *   second_prefix  : bytes[12]  merkle[28:32] + ntime + nbits     (changes with ntime)
 *   target_le      : bytes[32]  pool share target, little-endian
 *   soft_target_le : bytes[32]  local soft-share target, little-endian
 *   nonce_start    : int        inclusive start of this thread's nonce slice
 *   nonce_end      : int        exclusive end
 *
 * Returns a 3-tuple so Python can update stats without re-entering the loop.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <openssl/sha.h>
#include <string.h>
#include <stdint.h>

/* Compare a 32-byte hash to a 32-byte little-endian target.
 * Both are treated as 256-bit little-endian integers
 * (byte[31] is most significant). Returns 1 if hash < target. */
static inline int hash_lt_target(const uint8_t* hash, const uint8_t* target) {
    for (int i = 31; i >= 0; i--) {
        if (hash[i] < target[i]) return 1;
        if (hash[i] > target[i]) return 0;
    }
    return 0;  /* equal — not strictly less than */
}

static PyObject* mine_range(PyObject* self, PyObject* args) {
    Py_buffer fb, sp, tb, stb;
    unsigned long long nonce_start, nonce_end;

    if (!PyArg_ParseTuple(args, "y*y*y*y*KK",
                          &fb, &sp, &tb, &stb,
                          &nonce_start, &nonce_end))
        return NULL;

    PyObject* result = NULL;

    if (fb.len != 64 || sp.len != 12 || tb.len != 32 || stb.len != 32) {
        PyErr_SetString(PyExc_ValueError,
                        "buffer sizes must be 64, 12, 32, 32");
        goto cleanup;
    }

    {
        const uint8_t* first_block    = (const uint8_t*)fb.buf;
        const uint8_t* second_prefix  = (const uint8_t*)sp.buf;
        const uint8_t* target_le      = (const uint8_t*)tb.buf;
        const uint8_t* soft_target_le = (const uint8_t*)stb.buf;

        /* Compute SHA-256 midstate over the first 64-byte header block once.
         * SHA256_CTX is safe to memcpy in both OpenSSL 1.x and 3.x — the
         * struct layout (h[8], Nl, Nh, data[16], num, md_len) has not changed. */
        SHA256_CTX midstate;
        SHA256_Init(&midstate);
        SHA256_Update(&midstate, first_block, 64);

        /* second_block = second_prefix (12 bytes) + nonce (4 bytes) */
        uint8_t second_block[16];
        memcpy(second_block, second_prefix, 12);

        uint8_t inner[32], hash_out[32];

        uint64_t hashes_done = 0;
        uint64_t soft_shares = 0;
        long long  winner    = -1;

        for (uint64_t nonce = nonce_start; nonce < nonce_end; nonce++) {
            /* Pack nonce little-endian into last 4 bytes of second block */
            second_block[12] = (uint8_t)(nonce);
            second_block[13] = (uint8_t)(nonce >>  8);
            second_block[14] = (uint8_t)(nonce >> 16);
            second_block[15] = (uint8_t)(nonce >> 24);

            /* SHA-256d with midstate:
             *   1. Copy saved midstate (skips first 64-byte block compression)
             *   2. Process remaining 16 bytes + padding
             *   3. SHA-256 the 32-byte inner digest (outer pass) */
            SHA256_CTX ctx;
            memcpy(&ctx, &midstate, sizeof(SHA256_CTX));
            SHA256_Update(&ctx, second_block, 16);
            SHA256_Final(inner, &ctx);
            SHA256(inner, 32, hash_out);

            hashes_done++;

            if (hash_lt_target(hash_out, target_le)) {
                winner = (long long)nonce;
                break;
            }
            if (hash_lt_target(hash_out, soft_target_le)) {
                soft_shares++;
            }
        }

        if (winner >= 0)
            result = Py_BuildValue("(LKK)",
                                   winner, hashes_done, soft_shares);
        else
            result = Py_BuildValue("(OKK)",
                                   Py_None, hashes_done, soft_shares);
    }

cleanup:
    PyBuffer_Release(&fb);
    PyBuffer_Release(&sp);
    PyBuffer_Release(&tb);
    PyBuffer_Release(&stb);
    return result;
}

static PyMethodDef MinerCoreMethods[] = {
    {
        "mine_range", mine_range, METH_VARARGS,
        "mine_range(first_block, second_prefix, target_le, soft_target_le,\n"
        "           nonce_start, nonce_end)\n"
        "-> (winning_nonce | None, hashes_done, soft_shares)\n\n"
        "Runs SHA-256d Bitcoin mining loop entirely in C using OpenSSL midstate."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef miner_core_module = {
    PyModuleDef_HEAD_INIT, "miner_core",
    "Native SHA-256d mining inner loop — eliminates Python interpreter overhead.",
    -1, MinerCoreMethods
};

PyMODINIT_FUNC PyInit_miner_core(void) {
    return PyModule_Create(&miner_core_module);
}
