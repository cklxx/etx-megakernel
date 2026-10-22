import hashlib
from pathlib import Path

from etx.importer.launch_spec import CAPTURE_ROOT, verify_capture
from etx.importer.triton_import import import_triton

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'kernels/gfx942/mi300x/qwen3-30b-a3b/moe-up/vllm-triton-bf16'
EXPORT = 'fused_moe_up'


def test_moe_up_has_one_library_owned_copy():
    legacy = CAPTURE_ROOT / EXPORT
    assert KERNEL.is_dir() and not KERNEL.is_symlink()
    assert legacy.is_symlink() and legacy.resolve() == KERNEL.resolve()
    record = verify_capture(EXPORT)
    for name, expected in record['sha256'].items():
        artifact = KERNEL / name
        assert artifact.is_file() and not artifact.is_symlink()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    assert hashlib.sha256((KERNEL / 'CACHE_DIR').read_bytes()).hexdigest() == (
        '426a6cfd6bffe5290908a7bb8e54c0107accab27377b7ba7aa835f059573123d'
    )


def test_library_capture_import_matches_legacy_path():
    direct = import_triton(KERNEL, 'fused_moe_kernel', 'etx_imp_' + EXPORT, 1024, 1)
    legacy = import_triton(CAPTURE_ROOT / EXPORT, 'fused_moe_kernel', 'etx_imp_' + EXPORT, 1024, 1)
    assert direct == legacy
    source, info, meta, notes = direct
    assert '@etx_imp_fused_moe_up' in source
    assert info.block == (256, 1, 1)
    assert info.lds_bytes_per_slot == 20480
    assert info.arg_bytes > 0 and info.slots == 1
