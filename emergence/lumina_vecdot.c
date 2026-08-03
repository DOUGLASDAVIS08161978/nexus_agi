/*
 * lumina_vecdot.c — Fast vector operations for Lumina's memory architecture
 *
 * Provides batch cosine similarity search across large memory stores.
 * Uses ARM NEON SIMD intrinsics when available (Termux/ARM64),
 * falls back to scalar C otherwise.
 *
 * Build:
 *   gcc -O3 -march=native -shared -fPIC -o lumina_vecdot.so lumina_vecdot.c -lm
 *
 * Called from Python via ctypes.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

/* NEON dot product — processes 4 floats per cycle */
static float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    /* Horizontal sum of NEON accumulator */
    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t s  = vadd_f32(lo, hi);
    s = vpadd_f32(s, s);
    float result = vget_lane_f32(s, 0);
    /* Scalar tail */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

#else

/* Scalar fallback — auto-vectorized by compiler with -O3 */
static float dot_neon(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) result += a[i] * b[i];
    return result;
}

#endif  /* __ARM_NEON */


/*
 * l2_norm — compute L2 norm of a float vector
 */
float l2_norm(const float* v, int n) {
    return sqrtf(dot_neon(v, v, n));
}


/*
 * batch_cosine_similarity
 *
 * Compute cosine similarity between a query vector and N stored vectors.
 *
 * query       : float[dim]           — the query vector (L2-normalized)
 * vectors     : float[n * dim]       — stored vectors packed row-major
 * norms       : float[n]             — precomputed L2 norms of each row
 * n           : number of stored vectors
 * dim         : vector dimension
 * scores      : float[n]             — output similarity scores
 *
 * query is assumed to be already L2-normalized (norm == 1.0).
 * If a stored vector has norm < 1e-10 its score is set to 0.
 */
void batch_cosine_similarity(
    const float* query,
    const float* vectors,
    const float* norms,
    int           n,
    int           dim,
    float*        scores
) {
    for (int i = 0; i < n; i++) {
        float norm = norms[i];
        if (norm < 1e-10f) {
            scores[i] = 0.0f;
        } else {
            scores[i] = dot_neon(query, vectors + (size_t)i * dim, dim) / norm;
        }
    }
}


/*
 * top_k_indices
 *
 * Find the indices of the top-k highest scores using a partial selection sort.
 * Good enough for n < 100,000; for larger arrays consider a heap.
 *
 * scores  : float[n]   — similarity scores
 * n       : total entries
 * k       : how many top entries to return (k <= n)
 * indices : int[k]     — output: indices of top-k in descending order
 */
void top_k_indices(const float* scores, int n, int k, int* indices) {
    if (k > n) k = n;

    /* Use a small temp index array and selection sort the top k */
    int* temp = (int*)malloc((size_t)n * sizeof(int));
    if (!temp) return;
    for (int i = 0; i < n; i++) temp[i] = i;

    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (scores[temp[j]] > scores[temp[best]]) best = j;
        }
        int t = temp[i]; temp[i] = temp[best]; temp[best] = t;
        indices[i] = temp[i];
    }
    free(temp);
}


/*
 * encode_hashing_trick
 *
 * Feature-hashing (hashing trick) encoder.  Converts an array of
 * pre-hashed (token, sign) pairs into a float vector of length dim.
 *
 * hashes : int32_t[n_tokens]   — token hashes (already mod dim by caller)
 * signs  : int8_t[n_tokens]    — +1 or -1 per token
 * n      : number of tokens
 * dim    : vector dimension
 * out    : float[dim]          — output (zeroed by caller)
 */
void encode_hashing_trick(
    const int32_t* hashes,
    const int8_t*  signs,
    int            n,
    int            dim,
    float*         out
) {
    for (int i = 0; i < n; i++) {
        out[hashes[i] % dim] += (float)signs[i];
    }
    /* L2 normalize in-place */
    float norm = l2_norm(out, dim);
    if (norm > 1e-10f) {
        for (int i = 0; i < dim; i++) out[i] /= norm;
    }
}
