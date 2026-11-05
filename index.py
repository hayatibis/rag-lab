"""İndeksleme boru hattı: load → chunk → embed."""
import argparse
import sys

from rag import chunk, embed, load
from rag.embed import OllamaError

STEPS = ("load", "chunk", "embed")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dokümanları indeksle")
    p.add_argument("--step", choices=[*STEPS, "all"], default="all",
                   help="yalnız bir adımı çalıştır (öncekilerin çıktısı data/ altında olmalı)")
    args = p.parse_args(argv)
    todo = STEPS if args.step == "all" else (args.step,)
    try:
        if "load" in todo:
            load.run()
        if "chunk" in todo:
            chunk.run()
        if "embed" in todo:
            embed.run()
    except OllamaError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
