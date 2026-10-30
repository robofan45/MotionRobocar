"""
README checks: every table-of-contents link must point at a real heading,
every relative link or image must point at a file that exists, and every
algorithm section carries a "how it works" diagram (Mermaid, drawn by GitHub).

author: Mahim Trivedi
"""
import pathlib
import re

import pytest

import conftest  # noqa: F401  (adds the project root to sys.path)

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# overview diagrams: they name real folders (Name/) and files (name.py)
OVERVIEW_SECTIONS = ["How it works", "How testing works"]
SECTIONS_WITH_DIAGRAMS = OVERVIEW_SECTIONS + [
    "Localization", "Mapping", "SLAM", "Path Planning", "Path Tracking",
    "Arm Navigation", "Aerial Navigation", "Bipedal"]


def _github_slug(heading):
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("\\", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(text):
    headings = re.findall(r"^#+\s+(.*?)\s*$", text, flags=re.MULTILINE)
    anchors = {_github_slug(h) for h in headings}
    anchors |= set(re.findall(r'<a id="([^"]+)"', text))
    return anchors


def test_table_of_contents_links_point_at_headings():
    text = README.read_text(encoding="utf-8")
    anchors = _anchors(text)
    broken = [a for a in re.findall(r"\]\(#([^)]+)\)", text)
              if a not in anchors]
    assert not broken, f"README links to missing headings: {broken}"


def test_relative_links_and_images_exist():
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", text)
    targets += re.findall(r'src="([^"]+)"', text)
    missing = [t for t in targets
               if not re.match(r"[a-z]+:", t) and not (ROOT / t).exists()]
    assert not missing, f"README points at missing files: {missing}"


def _top_level_sections(text):
    """{title: text} for every '# Title' heading outside code blocks."""
    sections, title, in_fence = {}, None, False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# "):
            title = re.sub(r"<[^>]+>", "", line[2:]).strip()
            sections[title] = []
            continue
        if title is not None:
            sections[title].append(line)
    return {t: "\n".join(lines) for t, lines in sections.items()}


def _mermaid_blocks(text):
    return re.findall(r"^\s*```mermaid\s*\n(.*?)^\s*```\s*$", text,
                      flags=re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize("section", SECTIONS_WITH_DIAGRAMS)
def test_section_has_a_diagram(section):
    sections = _top_level_sections(README.read_text(encoding="utf-8"))
    assert section in sections, f"README has no '# {section}' section"
    assert _mermaid_blocks(sections[section]), \
        f"'# {section}' has no Mermaid diagram"


def test_diagrams_are_well_formed_flowcharts():
    """Cheap syntax guard (GitHub draws the diagrams, so a typo would show
    up as an error box instead of a picture)."""
    problems = []
    for n, block in enumerate(_mermaid_blocks(
            README.read_text(encoding="utf-8")), start=1):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not re.fullmatch(r"flowchart (TD|TB|LR|RL|BT)", lines[0]):
            problems.append(f"diagram {n}: first line must be "
                            f"'flowchart <direction>', got {lines[0]!r}")
        for ln in lines:
            if ln.count('"') % 2:
                problems.append(f"diagram {n}: unbalanced quote: {ln}")
                continue
            bare = re.sub(r'"[^"]*"', '""', ln)  # ignore text inside labels
            if any(bare.count(o) != bare.count(c)
                   for o, c in ("[]", "()", "{}")):
                problems.append(f"diagram {n}: unbalanced bracket: {ln}")
        opened = sum(ln.startswith("subgraph ") for ln in lines)
        closed = sum(ln == "end" for ln in lines)
        if opened != closed:
            problems.append(f"diagram {n}: {opened} subgraph(s) but "
                            f"{closed} 'end' line(s)")
    assert not problems, "\n".join(problems)


def test_overview_diagrams_name_real_folders_and_files():
    sections = _top_level_sections(README.read_text(encoding="utf-8"))
    named, missing = [], []
    for section in OVERVIEW_SECTIONS:
        for block in _mermaid_blocks(sections.get(section, "")):
            folders = re.findall(r"\b([A-Z][A-Za-z]+)/", block)
            files = re.findall(r"\b([\w-]+\.(?:py|sh))\b", block)
            named += folders + files
            missing += [f + "/" for f in folders if not (ROOT / f).is_dir()]
            missing += [f for f in files if not any(ROOT.rglob(f))]
    assert named, "overview diagrams should name the folders and files"
    assert not missing, f"diagrams name things that do not exist: {missing}"


if __name__ == '__main__':
    conftest.run_this_test(__file__)
