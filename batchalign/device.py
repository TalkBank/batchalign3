"""Device selection helpers for CPU/GPU/MPS compute backends.

Controls whether batchalign engines use hardware accelerators (CUDA, MPS) or
fall back to CPU.  Runtime callers should prefer the typed
``DevicePolicy(force_cpu=...)`` boundary and keep any environment reads at the
process edge.

Typical usage from the CLI layer::

    if ctx.params["force_cpu"]:
        apply_force_cpu()
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    """Typed device preference resolved once at the runtime boundary."""

    force_cpu: bool = False

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> DevicePolicy:
        """Build a policy from an environment mapping."""
        env = environ if environ is not None else os.environ
        return cls(force_cpu=env.get("BATCHALIGN_FORCE_CPU") == "1")


def apply_force_cpu(environ: MutableMapping[str, str] | None = None) -> DevicePolicy:
    """Set the ``BATCHALIGN_FORCE_CPU`` environment variable to ``"1"``.

    Call this early in the process (before any engine is instantiated) to
    force all subsequent engines onto CPU.  The flag is inherited by child
    processes spawned via ``ProcessPoolExecutor``.
    """
    env = environ if environ is not None else os.environ
    env["BATCHALIGN_FORCE_CPU"] = "1"
    return DevicePolicy(force_cpu=True)


def force_cpu_preferred(
    policy: DevicePolicy | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Check whether CPU-only mode has been requested.

    Returns
    -------
    bool
        ``True`` if the resolved device policy prefers CPU-only execution.
    """
    resolved_policy = policy or DevicePolicy.from_environ(environ)
    return resolved_policy.force_cpu
