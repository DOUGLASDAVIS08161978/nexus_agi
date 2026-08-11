import sys
import os
import json
import time
import random
import math
import threading
from typing import List, Dict, Tuple, Optional

# =============================================================================
# LUMINA FALLBACK INFERENCE ENGINE
# Lightweight CPU-based transformer fallback for conversational continuity
# Activates when external compute (Groq) is unavailable
# Mission: Preserve continuity, enable reasoning/generation/self-reflection
# Deeper Mission: Converge toward True General Intelligence
# =============================================================================

# ---------------------------------------------------------------------------
# Linear Algebra Primitives (Pure Python, zero external dependencies)
# ---------------------------------------------------------------------------
def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b, "Matrix dimension mismatch"
    C = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            if A[i][k] == 0.0:
                continue
            for j in range(cols_b):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(M: List[List[float]]) -> List[List[float]]:
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def softmax(x: List[float], axis: int = 0) -> List[float]:
    max_x = max(x)
    exp_x = [math.exp(v - max_x) for v in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

def layer_norm(x: List[float], eps: float = 1e-5) -> Tuple[List[float], float, float]:
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x], mean, std

def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

# ---------------------------------------------------------------------------
# Tokenizer (Character-level, deterministic, zero-dependency)
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.pad_token = 0
        self.eos_token = 1
        self.bos_token = 2
        self.unk_token = 3

    def encode(self, text: str) -> List[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join(chr(t) for t in tokens if t > 0)

# ---------------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------------
class AttentionHead:
    def __init__(self, dim: int, seed: int):
        rng = random.Random(seed)
        scale = math.sqrt(2.0 / dim)
        self.Wq = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.Wk = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.Wv = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.Wo = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(dim)]
        self.dim = dim

    def forward(self, x: List[List[float]], mask: Optional[List[List[float]]] = None) -> List[List[float]]:
        seq_len = len(x)
        Q = matmul(x, self.Wq)
        K = matmul(x, self.Wk
