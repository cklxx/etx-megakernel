"""CPU acceptance adapter; preserve runtime environment without recording secrets."""
from pathlib import Path

from etx.experiments import acceptance


def run_cpu_acceptance(output, *, formal=False, require_external=False, rocm_image=None,
                       rocm_root=None, local_matrix=False):
    # The internal supervisor intentionally inherits the caller's environment.
    # Passing os.environ through a persisted ProcessRequest would disclose secrets
    # and a filtered environment would change toolchain/prerequisite semantics.
    return acceptance.run(Path(output), formal=formal, require_external=require_external,
                          rocm_image=rocm_image,
                          rocm_root=Path(rocm_root) if rocm_root is not None else None,
                          local_matrix=local_matrix)


def cpu_acceptance_fixtures(output):
    return acceptance.fixtures(Path(output))
