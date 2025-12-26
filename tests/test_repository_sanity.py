from __future__ import annotations

from pathlib import Path


def test_docs_are_asciidoc() -> None:
    docs = list(Path("docs").glob("*"))
    assert docs, "docs/ is empty"
    assert all(p.suffix == ".adoc" for p in docs), f"Non-AsciiDoc docs found: {docs}"


def test_cli_script_present() -> None:
    p = Path("triton_contrast_cli.py")
    assert p.exists(), "triton_contrast_cli.py missing"
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith("#!/usr/bin/env python3"), "Expected shebang at top"
    assert "michelson_weber_triton" in txt, "Expected main metric fn name"
    # Ensure we didn't accidentally reformat/alter user code
    assert "Triton + Torch GPU contrast metrics" in txt


def test_readme_is_asciidoc() -> None:
    p = Path("README.adoc")
    assert p.exists()
    assert p.read_text(encoding="utf-8").lstrip().startswith("=")
