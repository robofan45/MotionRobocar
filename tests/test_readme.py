"""
README checks: every table-of-contents link must point at a real heading and
every relative link or image must point at a file that exists.

author: Mahim Trivedi
"""
import pathlib
import re

import conftest  # noqa: F401  (adds the project root to sys.path)

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


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


if __name__ == '__main__':
    conftest.run_this_test(__file__)
