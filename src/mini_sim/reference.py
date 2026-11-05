"""Untimed scalar oracle, independent of the array's reduction implementation.

Use the original dtype conversion utilities, but compute each dot product here.
NumPy matmul alone is not a bit-exact oracle for these pipelined adder trees.
"""

import numpy as np
from base.dtype import cast_vector, normalize_dtype


def reference_product(activations, weights, dtype="bf16", mixed=False):
    dt = normalize_dtype(dtype)

    def cast(row):
        return np.asarray(cast_vector(row, dt) if dt else row, dtype=np.float32)

    a = np.asarray([cast(row) for row in activations])
    w = np.asarray([cast(row) for row in weights])
    n = w.shape[0]
    log2n = n.bit_length() - 1
    groups = ([4] * (log2n // 2) + ([2] if log2n % 2 else [])) if mixed else [2] * log2n
    out = []
    for row in a:
        result = []
        for col in range(n):
            terms = [np.float32(row[k] * w[k, col]) for k in range(n)]
            for width in groups:
                terms = [np.float32(sum(float(x) for x in terms[i:i + width]))
                         for i in range(0, len(terms), width)]
            result.append(float(terms[0]))
        out.append(cast(result))
    return np.asarray(out, dtype=np.float32)

