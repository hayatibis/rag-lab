from pathlib import Path

import docx

from rag.load import docx_to_text, html_to_text, load_corpus, should_include


def test_should_include_filters():
    assert should_include(Path("notlar/README.md"))
    assert should_include(Path("rehber/kurulum.HTML"))
    assert should_include(Path("archive/proje/notes.md"))
    assert not should_include(Path("archive/old/a.md"))
    assert not should_include(Path("proje/.git/x.md"))
    assert not should_include(Path("rehber/kurulum-offline.html"))
    assert not should_include(Path("notlar/veri.json"))


def test_html_to_text_strips_noise_and_keeps_structure():
    html = """<html><head><title>T</title><style>.a{color:red}</style></head><body>
    <h1>Başlık</h1><p>İlk <b>paragraf</b>.</p><script>alert(1)</script>
    <img src="data:image/png;base64,AAAA"><svg><text>x</text></svg>
    <h2>Alt</h2><ul><li>bir</li><li>iki</li></ul>
    <table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
    </body></html>"""
    text = html_to_text(html)
    assert "# Başlık" in text
    assert "## Alt" in text
    assert "İlk paragraf." in text
    assert "- bir" in text and "- iki" in text
    assert "a | b" in text and "1 | 2" in text
    for junk in ("alert", "base64", "color:red", "<"):
        assert junk not in text


def test_html_table_row_starting_with_hash_is_not_a_heading():
    html = "<table><tr><th>#</th><th>Madde</th></tr><tr><td>1</td><td>x</td></tr></table>"
    lines = html_to_text(html).splitlines()
    assert "| # | Madde |" in lines
    assert not any(line.startswith("#") for line in lines)


def test_html_pre_is_fenced_so_comments_are_not_headings():
    text = html_to_text("<h1>Kurulum</h1><pre># FOO=bar\necho hi</pre>")
    assert "```\n# FOO=bar\necho hi\n```" in text


def test_html_source_newlines_are_spaces_not_line_breaks():
    text = html_to_text("<p>varsayılan başlığı altında\n# APP_MODEL=x diyor</p>")
    assert "altında # APP_MODEL=x diyor" in text
    assert not any(line.startswith("#") for line in text.splitlines())


def test_docx_headings_become_markdown(tmp_path):
    d = docx.Document()
    d.add_heading("Karar", level=1)
    d.add_paragraph("Gövde metni.")
    d.add_heading("Gerekçe", level=2)
    d.add_paragraph("Çünkü.")
    path = tmp_path / "a.docx"
    d.save(path)
    text = docx_to_text(path)
    assert "# Karar" in text
    assert "## Gerekçe" in text
    assert "Gövde metni.\n\n## Gerekçe" in text


def test_load_corpus_dedupes_reports_and_survives_bad_input(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nmetin", encoding="utf-8")
    (tmp_path / "kopya.md").write_text("# A\n\n   metin  ", encoding="utf-8")
    (tmp_path / "bos.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "latin.md").write_bytes(b"# Baslik\n\n\xff\xfe metin")
    (tmp_path / "bozuk.pdf").write_bytes(b"pdf degil")

    docs, report = load_corpus(tmp_path)

    assert [d.path for d in docs] == ["a.md", "latin.md"]
    assert "�" in docs[1].text
    assert any(r.startswith("KOPYA kopya.md") for r in report)
    assert any(r.startswith("BOŞ bos.md") for r in report)
    assert any(r.startswith("HATA bozuk.pdf") for r in report)
