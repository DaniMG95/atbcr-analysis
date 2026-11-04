"""Run independent CLI commands with bounded parallelism."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Result of one command from the batch."""

    number: int
    command: str
    returncode: int


def read_commands(path: Path) -> list[str]:
    """Read one command per line, ignoring blank lines and comments."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ValueError(f"No se pudo leer {path}: {error}") from error

    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


async def stream_output(number: int, stream: asyncio.StreamReader) -> None:
    """Print subprocess output with a stable command prefix."""
    while line := await stream.readline():
        print(f"[{number}] {line.decode(errors='replace').rstrip()}", flush=True)


async def run_command(
    number: int,
    command: str,
    *,
    cwd: Path,
    environment: dict[str, str],
) -> CommandResult:
    """Run a command and stream its combined output."""
    print(f"[{number}] INICIO: {command}", flush=True)
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None

    try:
        await stream_output(number, process.stdout)
        returncode = await process.wait()
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        raise

    status = "OK" if returncode == 0 else f"ERROR ({returncode})"
    print(f"[{number}] FIN: {status}", flush=True)
    return CommandResult(number, command, returncode)


async def run_batch(commands: list[str], *, jobs: int, cwd: Path) -> list[CommandResult]:
    """Consume commands through a fixed-size worker pool."""
    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    for item in enumerate(commands, start=1):
        queue.put_nowait(item)
    for _ in range(jobs):
        queue.put_nowait(None)

    environment = os.environ.copy()
    executable_dir = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = executable_dir + os.pathsep + environment.get("PATH", "")
    results: list[CommandResult] = []

    async def worker() -> None:
        while (item := await queue.get()) is not None:
            number, command = item
            try:
                results.append(
                    await run_command(number, command, cwd=cwd, environment=environment),
                )
            finally:
                queue.task_done()
        queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(jobs)]
    try:
        await queue.join()
        await asyncio.gather(*workers)
    except BaseException:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    return sorted(results, key=lambda result: result.number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta comandos independientes del CLI en paralelo.",
    )
    parser.add_argument(
        "commands_file",
        nargs="?",
        type=Path,
        help="archivo UTF-8 con un comando por linea",
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        default=[],
        help="comando adicional; se puede repetir",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="numero maximo de comandos simultaneos (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="directorio de trabajo de los comandos (por defecto: el actual)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="muestra los comandos sin ejecutarlos",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs debe ser mayor o igual que 1")

    commands = list(args.command)
    if args.commands_file is not None:
        try:
            commands = read_commands(args.commands_file) + commands
        except ValueError as error:
            parser.error(str(error))
    if not commands:
        parser.error("indica un archivo de comandos o al menos un --command")

    cwd = args.cwd.resolve()
    if not cwd.is_dir():
        parser.error(f"el directorio de trabajo no existe: {cwd}")

    print(f"Comandos: {len(commands)} | Paralelismo: {args.jobs} | Directorio: {cwd}")
    if args.dry_run:
        for number, command in enumerate(commands, start=1):
            print(f"[{number}] {command}")
        return 0

    try:
        results = asyncio.run(run_batch(commands, jobs=args.jobs, cwd=cwd))
    except KeyboardInterrupt:
        print("\nEjecucion cancelada.", file=sys.stderr)
        return 130

    failed = [result for result in results if result.returncode != 0]
    print(f"Resumen: {len(results) - len(failed)} correctos, {len(failed)} fallidos")
    if failed:
        for result in failed:
            print(
                f"  [{result.number}] codigo {result.returncode}: {result.command}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
