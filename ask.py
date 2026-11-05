"""Soru-cevap CLI. Soru verilmezse etkileşimli döngü açar."""
import argparse
import sys

from rag import config, trace
from rag.answer import SYSTEM_PROMPT, TRANSIENT_ERRORS, MissingApiKey, answer, build_prompt
from rag.embed import OllamaError
from rag.search import Hit, Index, IndexMissing, search


def format_sources(hits: list[Hit]) -> str:
    return "\n".join(
        f"  [{n}] {h.score:.3f}  {' › '.join([h.path, *h.heading_path])}"
        for n, h in enumerate(hits, 1)
    )


def run_query(index: Index, question: str, k: int, debug: bool, retrieve_only: bool) -> str:
    hits = search(index, question, k)
    out: list[str] = []
    if debug:
        for n, h in enumerate(hits, 1):
            out.append(f"--- [{n}] skor={h.score:.3f} {h.id}\n{h.text}\n")
    if retrieve_only:
        out.append("Getirilen parçalar:\n" + format_sources(hits))
        return "\n".join(out)
    if debug:
        out.append("=== Claude'a giden prompt ===\n[SYSTEM]\n" + SYSTEM_PROMPT
                   + "\n\n[USER]\n" + build_prompt(question, hits) + "\n=== prompt sonu ===\n")
    out.append(answer(question, hits))
    out.append("\nKaynaklar:\n" + format_sources(hits))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dokümanlara soru sor")
    p.add_argument("question", nargs="?", help="soru (verilmezse etkileşimli mod)")
    p.add_argument("-k", type=int, default=config.TOP_K, help="getirilecek parça sayısı")
    p.add_argument("--debug", action="store_true", help="parçaları ve prompt'u göster")
    p.add_argument("--retrieve-only", action="store_true", help="Claude'u çağırma, yalnız arama")
    p.add_argument("--trace", action="store_true",
                   help="her adımı görselleştiren HTML üret (data/traces/)")
    args = p.parse_args(argv)

    try:
        index = Index.load()
    except IndexMissing as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    def ask_once(q: str) -> bool:
        try:
            if args.trace:
                t = trace.run_traced(index, q, args.k, retrieve_only=args.retrieve_only)
                hits = [Hit(h["id"], h["path"], h["heading_path"], h["score"], h["text"])
                        for h in t["search"]["hits"]]
                if t["generate"]:
                    print(t["generate"]["answer"])
                print("\nKaynaklar:\n" + format_sources(hits))
                print(f"\nİz: {trace.save(t)}")
            else:
                print(run_query(index, q, args.k, args.debug, args.retrieve_only))
            return True
        except (OllamaError, MissingApiKey, *TRANSIENT_ERRORS) as e:
            print(f"HATA: {type(e).__name__}: {e}", file=sys.stderr)
            return False

    if args.question:
        return 0 if ask_once(args.question) else 1

    print(f"{len(index.chunks)} parça yüklü. Çıkmak için boş satır ya da Ctrl-D.")
    while True:
        try:
            q = input("\nsoru> ").strip()
        except EOFError:
            break
        if not q:
            break
        ask_once(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
