"""Inputs for the byte-ledger command: the packed-manifest text a config implies (no weights)."""
from pathlib import Path


def packed_manifest_text(config):
    """The vl.manifest text examples/vllm_llm/byte_ledger.py derives for a Hugging Face config file."""
    from examples.vllm_llm import byte_ledger
    return byte_ledger.manifest_from_config(Path(config))
