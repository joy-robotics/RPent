"""JoyBrain-compatible msgpack NumPy wire format, without importing the ML stack."""

from __future__ import annotations

import math

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise TypeError(f"Unsupported NumPy dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(order="C"),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" not in value and b"__npgeneric__" not in value:
        return value
    dtype = np.dtype(value[b"dtype"])
    if dtype.kind not in "biuf":
        raise ValueError(f"Unsupported wire dtype: {dtype}")
    if b"__ndarray__" not in value:
        return dtype.type(value[b"data"])
    shape = value[b"shape"]
    if not isinstance(shape, (list, tuple)) or not all(
        type(size) is int and size >= 0 for size in shape
    ):
        raise ValueError("Invalid ndarray shape")
    data = value[b"data"]
    if not isinstance(data, bytes) or math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError("ndarray shape/dtype does not match payload size")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def pack(value) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(data: bytes):
    if not isinstance(data, bytes):
        raise ValueError(f"Expected binary msgpack, got {type(data).__name__}")
    return msgpack.unpackb(data, object_hook=_decode, raw=False, strict_map_key=False)
