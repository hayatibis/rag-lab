from rag.chunk import Section, chunk_doc, merge_short, split_long, split_sections


def test_split_sections_tracks_heading_path_and_ignores_code_fences():
    text = (
        "giriş\n\n# A\nmetin a\n## B\nmetin b\n"
        "```\n# yorum, başlık değil\n```\n"
        "### C\nmetin c\n# D\nmetin d"
    )
    secs = split_sections(text)
    assert [s.heading_path for s in secs] == [[], ["A"], ["A", "B"], ["A", "B", "C"], ["D"]]
    assert "# yorum, başlık değil" in secs[2].body
    assert secs[1].body.startswith("# A")


def test_split_long_respects_limit_and_overlaps():
    paras = [f"paragraf {i} " + "kelime " * 40 for i in range(10)]
    pieces = split_long("\n\n".join(paras), max_chars=1000, overlap_chars=100)
    assert len(pieces) > 1
    assert all(len(p) <= 1000 + 100 + 2 for p in pieces)
    for prev, cur in zip(pieces, pieces[1:]):
        head = cur.split("\n\n")[0]
        assert prev.endswith(head)


def test_split_long_hard_splits_giant_paragraph():
    pieces = split_long("x" * 2500, max_chars=1000, overlap_chars=0)
    assert [len(p) for p in pieces] == [1000, 1000, 500]


def test_split_long_prefers_word_boundary():
    pieces = split_long(("kelime " * 300).strip(), max_chars=1000, overlap_chars=0)
    assert all(not p.startswith("elime") for p in pieces)
    assert all(len(p) <= 1000 for p in pieces)


def test_merge_short_joins_sections_under_same_top_heading():
    secs = [Section(["A"], "# A\nkısa"), Section(["A", "B"], "## B\nkısa"), Section(["C"], "# C\nkısa")]
    merged = merge_short(secs, max_chars=1000)
    assert [m.heading_path for m in merged] == [["A"], ["C"]]
    assert "## B" in merged[0].body


def test_merge_short_respects_limit():
    secs = [Section(["A"], "a" * 600), Section(["A", "B"], "b" * 600)]
    assert len(merge_short(secs, max_chars=1000)) == 2


def test_chunk_doc_prefixes_context_label():
    chunks = chunk_doc("notlar/02.md", "# Karar\n## Gerekçe\nÇünkü böyle.")
    assert len(chunks) == 1
    assert chunks[0].id == "notlar/02.md#0"
    assert chunks[0].text.startswith("[notlar/02.md › Karar]\n")
    assert "Çünkü böyle." in chunks[0].text
