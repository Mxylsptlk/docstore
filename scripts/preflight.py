"""Preflight checks for docstore.

Run before a real ingest: verifies the environment is ready (Anthropic key, Ollama +
embedding model, writable data dir). Prints actionable fixes and exits non-zero on
failure.

Usage: python scripts/preflight.py [--config config.yaml]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docstore.config import Config, load_config  # noqa: E402


def check(cfg: Config) -> list[str]:
    problems: list[str] = []

    # 1. Anthropic key (only needed for the claude backend)
    if cfg.extraction_backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY is not set. Export it: export ANTHROPIC_API_KEY=sk-...")

    # 2. Ollama reachable + required models present (embeddings, and the vision model
    #    when the ollama vision backend is active).
    needed = [cfg.embed_model]
    if cfg.extraction_backend == "ollama":
        needed.append(cfg.vision_model_ollama)
    try:
        import ollama

        models = ollama.list().get("models", [])
        names = {m.get("model") or m.get("name") for m in models}
        for want in needed:
            if not any((want in (n or "")) for n in names):
                problems.append(
                    f"Ollama is up but model {want!r} is not pulled. Run: ollama pull {want}"
                )
    except Exception as e:  # noqa: BLE001
        pulls = " && ".join(f"ollama pull {m}" for m in needed)
        problems.append(
            f"Cannot reach Ollama ({e}). Start it with `ollama serve` and `{pulls}`."
        )

    # 3. Writable data dir
    try:
        p = Path(cfg.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except Exception as e:  # noqa: BLE001
        problems.append(f"data_dir {cfg.data_dir!r} is not writable: {e}")

    return problems


def main() -> int:
    config_path = None
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    cfg = load_config(config_path) if config_path else (
        load_config("config.yaml") if Path("config.yaml").exists() else Config()
    )

    problems = check(cfg)
    if problems:
        print("Preflight FAILED:\n")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("Preflight OK — environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
