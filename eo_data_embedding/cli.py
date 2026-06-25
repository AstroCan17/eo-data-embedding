"""eo-data-embedding command-line interface.

`eo-data-embedding demo` is the plug-and-play CPU demo and works from any install. The phase
subcommands (extract/search/probe/...) are thin pass-throughs to the scripts in `scripts/`, so they
only resolve in a source checkout (`git clone` + `pip install -e .`).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_PHASE_SCRIPTS = {
    "sanity": "phase0_sanity.py",
    "smoke": "phase0_smoke.py",
    "extract": "phase1_extract.py",
    "search": "phase2_search.py",
    "probe": "phase3_probe.py",
    "change": "phase5b_change_probe.py",
}

_HELP = """eo-data-embedding — multi-modal geospatial embedding toolkit

usage: eo-data-embedding <command> [args]

commands:
  demo     plug-and-play CPU demo (downloads EuroSAT + bundle, serves the UI) — no GPU
  app      serve the demo UI over an already-fetched bundle
  extract  phase 1 — embed a dataset with Clay (needs GPU/Clay)
  search   phase 2 — FAISS retrieval metrics
  probe    phase 3 — few-shot linear probe
  change   phase 5b — OSCD change-detection probe
  sanity   phase 0 — sanity check
  smoke    phase 0 — green-light smoke gate

run 'eo-data-embedding <command> --help' for a command's options.
"""


def _run_script(name: str, argv: list[str]) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / name
    if not script.exists():
        print(
            f"'{name}' not found — phase subcommands need a source checkout "
            "(git clone + pip install -e .). The 'demo' subcommand works from any install.",
            file=sys.stderr,
        )
        return 1
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code or 0)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "demo":
        from .demo import main as demo_main

        return demo_main(rest)
    if cmd == "app":
        from .demo import serve

        serve()
        return 0
    if cmd in _PHASE_SCRIPTS:
        return _run_script(_PHASE_SCRIPTS[cmd], rest)

    print(f"unknown command: {cmd}\n", file=sys.stderr)
    print(_HELP, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
