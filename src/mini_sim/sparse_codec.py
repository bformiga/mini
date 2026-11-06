"""Explicit prototype 2:4 encoding: values followed by packed two-bit indices."""

import numpy as np
from base.dtype import bf16_round


def compress_2of4(weights):
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] % 4 or not np.isfinite(w).all():
        raise ValueError("2:4 weights must be finite W[K,N] with K divisible by four")
    k, n = w.shape
    groups = w.reshape(k//4,4,n)
    present = groups != 0
    if np.any(present.sum(axis=1)>2):
        raise ValueError("weights violate 2:4 along K; prune explicitly before sparse simulation")
    positions = np.argsort(~present,axis=1,kind="stable")[:,:2,:]
    positions.sort(axis=1)
    values = np.take_along_axis(groups,positions,axis=1).reshape(k//2,n)
    indices = positions.reshape(k//2,n).astype(np.uint8)
    return values, indices


def validate_compressed(values, indices):
    v, ix = np.asarray(values), np.asarray(indices)
    if v.ndim != 2 or v.shape[0] % 2 or ix.shape != v.shape or not np.isfinite(v).all():
        raise ValueError("compressed values/indices must have matching [even_rows,N] shapes")
    if not np.issubdtype(ix.dtype, np.integer) or np.any(ix < 0) or np.any(ix > 3):
        raise ValueError("indices must be integers in 0..3")
    if np.any(ix[0::2] == ix[1::2]):
        raise ValueError("two retained positions in a group must be distinct")
    return v.astype(np.float32), ix.astype(np.uint8)


def decompress_2of4(values, indices):
    v, ix = validate_compressed(values, indices)
    rows, n = v.shape
    w = np.zeros((2*rows, n), dtype=np.float32)
    addresses = 4*(np.arange(rows)[:,None]//2)+ix
    w[addresses,np.arange(n)[None,:]] = v
    return w


def encode_numbers(values, dtype):
    a = np.asarray(values, dtype=np.float32)
    if dtype == "bf16":
        return (bf16_round(a).view(np.uint32) >> 16).astype("<u2").tobytes()
    return a.astype("<f2" if dtype == "fp16" else "<f4").tobytes()


def decode_numbers(payload, shape, dtype):
    if dtype == "bf16":
        a = (np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(payload, dtype="<f2" if dtype == "fp16" else "<f4").astype(np.float32)
    return a.reshape(shape)


def pack_sparse(values, indices, dtype):
    v, ix = validate_compressed(values, indices)
    # Column-major vectors match the weight shift interface. Four indices/byte.
    flat = ix.T.ravel()
    meta = np.zeros((flat.size + 3)//4, dtype=np.uint8)
    for shift in range(4):
        part = flat[shift::4]
        meta[:len(part)] |= part << (2*shift)
    return encode_numbers(v.T, dtype) + meta.tobytes()


def unpack_sparse(payload, shape, dtype):
    count = int(np.prod(shape))
    value_bytes = count * (4 if dtype == "fp32" else 2)
    if len(payload) != value_bytes + (count+3)//4:
        raise ValueError("incorrect compressed payload length")
    v = decode_numbers(payload[:value_bytes], (shape[1], shape[0]), dtype).T
    meta = np.frombuffer(payload[value_bytes:], dtype=np.uint8)
    flat = np.asarray([(meta[i//4] >> (2*(i%4))) & 3 for i in range(count)], dtype=np.uint8)
    ix = flat.reshape(shape[1], shape[0]).T
    return validate_compressed(v, ix)

