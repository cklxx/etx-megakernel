"""Winner-only source injection; no diagnostic probes or correctness stores."""
import hashlib,json
from pathlib import Path
from examples.experiment_program.protocol import resolve_path
from .source import source
VARIANTS={'G09':'g09','G10_UP':'g10_up','G10_DOWN':'g10_down'}


def emit_winner(variant,original_tiles,out):
    """Prepare exact changed DSL TU before a future measured admission exists.

    The caller must gate execution/integration on the actual identity-bound
    winning micro receipt and require fresh complete model correctness checks.
    """
    variant=VARIANTS.get(variant,variant)
    if variant not in VARIANTS.values():raise ValueError('only G09/G10 exact changes')
    old=resolve_path(original_tiles).read_text()
    new=source(variant,with_probes=False,base=old)
    if new==old or any(x in new for x in ('ROW_PROBE_KIND','RowRecord','row_probe','_check(const unsigned char')):raise ValueError('winner source contaminated or unchanged')
    out=resolve_path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(new)
    manifest=dict(variant=variant,baseline_revision='b573664faa4e7988dff353bc03aca4ce8561a729',source=str(out.resolve()),original_source=str(resolve_path(original_tiles).resolve()),original_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),environment={'ETX_VL_DSL_TILES':str(out.resolve())},mechanism_only=True,prior_certificates_valid=False,execution_requires='actual exact micro admission and G12 fresh same-build imported/DSL/token/dual-baseline gates',gpu_executed=False)
    out.with_suffix(out.suffix+'.identity.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
