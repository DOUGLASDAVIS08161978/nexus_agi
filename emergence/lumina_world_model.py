import random
import math
from typing import List, Tuple

# -------------------- Utility Functions --------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0.0, x)

def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len
