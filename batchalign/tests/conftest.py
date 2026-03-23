# Package-level conftest for batchalign tests.
#
# Shared fixtures live here.  For test doubles see doubles.py.
#
# SAFETY: OOM prevention hooks below enforce that golden/ML tests NEVER run
# with parallel xdist workers on machines with < 128 GB RAM. This prevents
# kernel-level OOM panics caused by multiple Stanza model instances (2-5 GB
# each) running concurrently.
#
# Incidents: 2026-03-19 (nextest parallel), 2026-03-23 (pytest -n 3 golden).

from __future__ import annotations

import platform
import pytest


def _get_system_ram_gb() -> int:
    """Return total system RAM in GB (macOS and Linux)."""
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) // (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024 * 1024)
    except Exception:
        pass
    return 0  # Unknown — be conservative


# Threshold: machines with less than this MUST NOT run golden tests in parallel.
_SAFE_RAM_GB = 128


def pytest_configure(config: pytest.Config) -> None:
    """Block parallel golden test execution on machines with insufficient RAM.

    This hook fires before test collection. If the user requested golden tests
    (via -m golden) and xdist parallelism is active (-n > 0), either force
    sequential execution or abort with a clear error.
    """
    ram_gb = _get_system_ram_gb()

    # Detect if xdist is active and how many workers
    num_workers = getattr(config.option, "numprocesses", None)
    if num_workers is None:
        # xdist not installed or not active
        return

    # num_workers can be "auto" string or int
    if isinstance(num_workers, str):
        if num_workers == "auto":
            import os
            num_workers = os.cpu_count() or 1
        else:
            try:
                num_workers = int(num_workers)
            except ValueError:
                return

    if num_workers <= 0:
        # -n 0 means no parallelism — safe
        return

    # Check if golden tests are being INCLUDED (not excluded).
    # Default pytest.ini uses "not slow and not golden and not integration" which
    # EXCLUDES golden. We only need protection when golden is actively selected.
    markexpr = config.option.markexpr or ""
    # "golden" included but NOT preceded by "not " — means golden tests are selected
    has_golden = "golden" in markexpr and "not golden" not in markexpr

    if has_golden and ram_gb > 0 and ram_gb < _SAFE_RAM_GB:
        # FORCE sequential execution — do not allow parallel golden on small machines
        config.option.numprocesses = 0
        config.option.dist = "no"

        import warnings
        warnings.warn(
            f"\n\n"
            f"  OOM PROTECTION: Forced -n 0 for golden tests.\n"
            f"  This machine has {ram_gb} GB RAM (< {_SAFE_RAM_GB} GB threshold).\n"
            f"  Each Stanza model instance uses 2-5 GB. Parallel workers would OOM.\n"
            f"  To run golden tests in parallel, use a machine with >= {_SAFE_RAM_GB} GB RAM.\n",
            stacklevel=1,
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Second safety net: if ANY collected test has the golden marker and we're
    running parallel, abort before execution starts."""
    num_workers = getattr(config.option, "numprocesses", None)

    # Normalize num_workers to int
    if num_workers is not None and not isinstance(num_workers, int):
        try:
            num_workers = int(num_workers)
        except (ValueError, TypeError):
            import os
            num_workers = os.cpu_count() or 1

    if num_workers is None or num_workers <= 0:
        return

    ram_gb = _get_system_ram_gb()
    if ram_gb >= _SAFE_RAM_GB:
        return

    golden_tests = [item for item in items if item.get_closest_marker("golden")]
    if golden_tests:
        pytest.exit(
            f"REFUSED: {len(golden_tests)} golden test(s) collected with "
            f"-n {num_workers} on a {ram_gb} GB machine. "
            f"Golden tests load ML models (2-5 GB each) and WILL cause OOM with "
            f"parallel workers. Use -n 0 or run on a machine with >= {_SAFE_RAM_GB} GB RAM.",
            returncode=1,
        )


@pytest.fixture(autouse=True)
def _guard_golden_oom(request: pytest.FixtureRequest) -> None:
    """Per-test OOM guard: refuse to run golden tests on low-RAM machines
    when parallel workers are active.

    This is a belt-and-suspenders guard that fires even if the collection
    hooks are bypassed by xdist worker distribution. Each golden test
    checks its own safety before loading any models.
    """
    marker = request.node.get_closest_marker("golden")
    if marker is None:
        return

    ram_gb = _get_system_ram_gb()
    if ram_gb >= _SAFE_RAM_GB:
        return

    # Check if we're running inside an xdist worker
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", None)
    if worker_id is not None:
        # We're in an xdist worker — this means parallel execution is active
        # on a machine that's too small. Fail the test immediately.
        pytest.fail(
            f"OOM PROTECTION: golden test '{request.node.name}' running in "
            f"xdist worker {worker_id} on a {ram_gb} GB machine. "
            f"Each worker loads its own ML models (2-5 GB). "
            f"Use -n 0 for golden tests on machines with < {_SAFE_RAM_GB} GB RAM."
        )
