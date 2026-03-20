"""CLI entry point for the inference worker process.

This module is intentionally thin. It owns only three responsibilities:

1. parse command-line arguments from the Rust launcher
2. configure logging for the worker process lifetime
3. delegate model/bootstrap decisions to the worker loading helpers

Keeping the entrypoint small makes it clear that orchestration policy lives in
Rust and model-loading policy lives in dedicated worker helper modules rather
than in an oversized `main()`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batchalign.inference._domain_types import TcpPort

from batchalign.device import DevicePolicy
from batchalign.worker._model_loading import (
    enable_test_echo,
    load_worker_profile,
    load_worker_task,
    parse_engine_overrides,
    resolve_injected_revai_api_key,
)
from batchalign.worker._types import InferTask, WorkerProfile
from batchalign.worker._protocol import (
    _print_ready,
    _serve_stdio,
    _serve_stdio_concurrent,
    _serve_tcp,
    _serve_tcp_concurrent,
)
from batchalign.worker._types import WorkerBootstrapRuntime, _state


def build_arg_parser():
    """Build the internal worker CLI parser used by the Rust launcher."""
    import argparse

    parser = argparse.ArgumentParser(description="Batchalign worker process")
    parser.add_argument("--task", default="", help="Infer task bootstrap target")
    parser.add_argument("--lang", default="eng", help="Language code")
    parser.add_argument("--num-speakers", type=int, default=1)
    parser.add_argument("--engine-overrides", default="", help="JSON dict of engine overrides")
    parser.add_argument(
        "--test-echo",
        action="store_true",
        help="Test mode: echo input unchanged (no ML models)",
    )
    parser.add_argument(
        "--test-delay-ms",
        type=int,
        default=0,
        help="Test mode: sleep this many ms before each response (for timeout testing)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (0=warn, 1=info, 2=debug, 3=trace)",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Worker profile (gpu, stanza, io) — groups related tasks into one process",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "tcp"],
        default="stdio",
        help="IPC transport: stdio (child process) or tcp (persistent daemon)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port to listen on (0 = auto-assign from 9100-9199). Only used with --transport tcp.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP bind address (default: 127.0.0.1). Only used with --transport tcp.",
    )
    return parser


def build_worker_bootstrap_runtime(
    args,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorkerBootstrapRuntime:
    """Resolve one typed worker bootstrap runtime from CLI args + boundary env."""
    env = environ if environ is not None else os.environ
    engine_overrides = parse_engine_overrides(args.engine_overrides) or {}
    task = None
    if args.task:
        try:
            task = InferTask(args.task)
        except ValueError as error:
            raise ValueError(f"unknown infer task: {args.task}") from error
    profile = None
    if args.profile:
        try:
            profile = WorkerProfile(args.profile)
        except ValueError as error:
            raise ValueError(f"unknown worker profile: {args.profile}") from error
    return WorkerBootstrapRuntime(
        task=task,
        lang=args.lang,
        num_speakers=args.num_speakers,
        profile=profile,
        engine_overrides=engine_overrides,
        test_echo=args.test_echo,
        verbose=args.verbose,
        device_policy=DevicePolicy(force_cpu=args.force_cpu),
        revai_api_key=resolve_injected_revai_api_key(env),
    )


def parse_worker_args(argv: Sequence[str] | None = None):
    """Parse worker CLI arguments into the raw argparse namespace."""
    return build_arg_parser().parse_args(argv)


def main() -> None:
    """Run the stdio worker bootstrap path used by the Rust server.

    The Rust side launches one worker process per infer-task/language
    combination. This function parses that launch contract, delegates setup to
    the model loader, then hands off to the long-lived stdio protocol loop.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        args.verbose, logging.DEBUG
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        bootstrap = build_worker_bootstrap_runtime(args)
    except ValueError as error:
        parser.error(str(error))  # pragma: no cover - argparse exits
    _state.bootstrap = bootstrap

    if args.test_echo:
        if bootstrap.profile is not None:
            label = f"profile:{bootstrap.profile.value}"
        elif bootstrap.task is not None:
            label = bootstrap.task.value
        else:
            label = "test-echo"
        enable_test_echo(label, bootstrap.lang)
        if args.test_delay_ms > 0:
            _state.test_delay_ms = args.test_delay_ms
    elif bootstrap.profile is not None:
        load_worker_profile(bootstrap)
    elif bootstrap.task is not None:
        load_worker_task(bootstrap)
    else:
        parser.error("--task or --profile is required (or use --test-echo)")

    # After bootstrap succeeds, the worker switches to the request loop
    # expected by the Rust supervisor (stdio) or becomes a persistent daemon
    # (TCP). GPU profile workers use concurrent serving (ThreadPoolExecutor)
    # so multiple requests can share loaded GPU models via in-process threads.
    if args.transport == "tcp":
        port = args.port if args.port != 0 else _auto_assign_port(args.host)
        if bootstrap.profile == WorkerProfile.GPU:
            _serve_tcp_concurrent(args.host, port)
        else:
            _serve_tcp(args.host, port)
    else:
        _print_ready()
        if bootstrap.profile == WorkerProfile.GPU:
            _serve_stdio_concurrent()
        else:
            _serve_stdio()


def _auto_assign_port(host: str) -> TcpPort:
    """Find an available port in the 9100-9199 range.

    Checks the worker registry to avoid collisions with already-registered
    workers, then verifies the port is bindable.
    """
    import socket

    from batchalign.worker._registry import list_workers

    registered = list_workers()
    used_ports = {e.port for e in registered if e.host == host}

    for candidate in range(9100, 9200):
        if candidate in used_ports:
            continue
        # Verify the port is actually available by trying to bind.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                return candidate
        except OSError:
            continue

    raise RuntimeError(
        f"No available ports in range 9100-9199 on {host}. "
        "Stop some workers or specify --port explicitly."
    )
