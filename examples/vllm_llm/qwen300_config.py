"""Shared Qwen300 configuration, without importing either session's budget guard."""
import math
import os

IMAGE = 'vllm/vllm-openai-rocm@sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
QWEN300_MAX_SESSION_SECONDS = 110 * 60
QWEN300_HOUR_MAX_SESSION_SECONDS = 8400


def validate_cap(maximum, name):
    message = f'{name} cap must be positive and <={maximum} seconds'
    try:
        seconds = float(os.environ.get('ETX_SESS_MAX_SECONDS', maximum))
    except ValueError:
        raise ValueError(message) from None
    if not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise ValueError(message)
