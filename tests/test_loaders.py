import pytest

from safety_rag.config import Source
from safety_rag.loaders import load_file, load_sources


def test_latex_headings_become_markdown(tmp_path):
    path = tmp_path / "chapter.tex"
    path.write_text(
        "% a comment that should vanish\n"
        "\\chapter{Implementation}\n"
        "\\section{Fall Detection}\n"
        "A fall is confirmed after \\textbf{consecutive} frames \\cite{smith2020}.\n"
        "\\begin{figure}\\includegraphics{x.png}\\end{figure}\n"
    )
    text = load_file(path, "report", "prose").text

    assert "# Implementation" in text
    assert "## Fall Detection" in text
    assert "consecutive" in text
    assert "a comment that should vanish" not in text
    assert "includegraphics" not in text
    assert "smith2020" not in text


def test_latex_escapes_do_not_survive(tmp_path):
    path = tmp_path / "escapes.tex"
    path.write_text("Event type PPE\\_VIOLATION is logged as ``UNIDENTIFIED'' when unknown.\n")
    text = load_file(path, "report", "prose").text

    assert "PPE_VIOLATION" in text
    assert "\\_" not in text
    assert "``" not in text and "''" not in text


def test_unsupported_suffix_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_file(path, "x", "prose")


def test_failures_are_collected_not_raised(tmp_path):
    (tmp_path / "good.md").write_text("# Title\n\nUsable body text for the index.\n")
    (tmp_path / "empty.md").write_text("")
    source = Source(label="s", root=tmp_path, include=("*.md",))

    docs, failures = load_sources([source])

    assert [d.path.name for d in docs] == ["good.md"]
    assert [f.path.name for f in failures] == ["empty.md"]


def test_missing_source_root_is_reported(tmp_path):
    source = Source(label="gone", root=tmp_path / "absent", include=("*.md",))
    docs, failures = load_sources([source])
    assert docs == []
    assert "source root missing" in failures[0].reason


def test_exclude_patterns_are_honoured(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "keep.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "skip.py").write_text("y = 2\n")
    source = Source(label="s", root=tmp_path, include=("pkg/*.py",), exclude=("pkg/skip.py",))
    assert [p.name for p in source.iter_files()] == ["keep.py"]
