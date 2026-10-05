"""Build-output contract tests for the docs_html portal in GPT-OSS-Lite.

Runs the real generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, ASCII hero, mono-only fonts, widget
containers. Also asserts structural integrity (balanced tags, resolvable local
links) and the accessibility landmarks. Markdown sources are never modified.
"""

import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_docs_html.py"
OUT = ROOT / "docs_html"

# Copied verbatim by the build, so not subject to the renderer's markup.
STATIC_HTML = {
    "docs/gpt_oss_visual_guide.html",
    "docs/gpt-oss-lite-model-architecture.html",
    "docs/gpt-oss-lite-optimization-stack.html",
    "docs/gpt-oss-lite-data-pipeline.html",
    "docs/gpt-oss-lite-training-workflow.html",
}

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, str(BUILD)], check=True, cwd=ROOT)
    return OUT


def read(rel: str) -> str:
    return (OUT / rel).read_text(encoding="utf-8")


def generated_pages(out: Path) -> list[Path]:
    """Every page the renderer writes, excluding copied static assets."""
    return [
        p for p in sorted(out.rglob("*.html"))
        if p.relative_to(out).as_posix() not in STATIC_HTML
    ]


class _BalanceChecker(HTMLParser):
    """Collects unbalanced-tag problems for one document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_ELEMENTS:
            self.stack.append((tag, self.getpos()))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID_ELEMENTS:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
            return
        names = [t for t, _ in self.stack]
        if tag not in names:
            self.problems.append(f"stray </{tag}> at line {self.getpos()[0]}")
            return
        index = len(names) - 1 - names[::-1].index(tag)
        for name, pos in reversed(self.stack[index + 1:]):
            self.problems.append(f"unclosed <{name}> opened at line {pos[0]}")
        self.stack = self.stack[:index]


def test_portal_js_asset_copied(built):
    assert (OUT / "assets" / "portal.js").is_file()


def test_doc_page_boot_wiring(built):
    html = read("README.html")
    assert 'id="boot-overlay"' in html
    assert "booting" in html
    assert '<script defer src="./assets/portal.js"></script>' in html


def test_nested_page_rel_prefix(built):
    html = read("docs/concepts/foundations-and-architecture.html")
    assert 'src="../../assets/portal.js"' in html


def test_font_link_mono_only(built):
    html = read("README.html")
    assert "IBM+Plex+Mono" in html
    assert "IBM+Plex+Serif" not in html


def test_index_hero_ascii_stage(built):
    html = read("index.html")
    assert 'id="hero-decode"' in html
    assert 'data-title="OPENAI-GPT-OSS"' in html
    assert "bottleneck-svg" not in html
    assert "hero-title sr-only" in html


def test_index_pass_widget_container(built):
    html = read("index.html")
    assert 'id="passWidget"' in html
    assert "passDiagramCanvas" in html


def test_moe_playground_container(built):
    assert 'id="widget-moe-routing"' in read("docs/concepts/moe.html")


def test_sink_toggle_container(built):
    assert 'id="widget-sink-bias"' in read("docs/concepts/attention-sinks.html")


def test_widgets_not_on_other_pages(built):
    html = read("docs/concepts/tokenization.html")
    assert "widget-moe-routing" not in html
    assert "widget-sink-bias" not in html
    assert "widget-layer-stack" not in html


def test_dark_theme_only(built):
    html = read("README.html")
    assert "toggleTheme" not in html
    assert "theme-toggle" not in html
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert '[data-theme="light"]' not in css


# --- structural integrity -------------------------------------------------

def test_all_manifest_docs_have_twins(built):
    """Every DOC_FILES entry must resolve to a real markdown source."""
    source = (ROOT / "scripts" / "build_docs_html.py").read_text(encoding="utf-8")
    manifest = re.findall(r'\("([^"]+\.md)", "(?:Core|Concepts|Guides|References)"', source)
    assert manifest, "no manifest entries parsed"
    for rel in manifest:
        assert (ROOT / rel).is_file(), f"manifest lists missing source: {rel}"
        assert (built / rel.replace(".md", ".html")).is_file(), f"no twin for {rel}"


def test_no_broken_local_links(built):
    """Local hrefs and srcs must resolve inside the built site."""
    broken = []
    for page in built.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r'(?:href|src)="([^"]+)"', text):
            url = match.group(1).split("#")[0].split("?")[0]
            if not url or url.startswith(("http://", "https://", "mailto:", "data:", "//")):
                continue
            if not (page.parent / url).exists():
                broken.append(f"{page.relative_to(built)} -> {url}")
    assert not broken, "broken local links:\n" + "\n".join(broken)


def test_source_links_resolve_on_detached_head(built, tmp_path, monkeypatch):
    """Regression: actions/checkout materialises a PR as a detached merge ref,
    where `git branch --show-current` is empty. The build must still resolve a
    GitHub base (via the commit SHA) instead of leaving source links relative
    into docs_html/, which made CI fail the link check."""
    import subprocess as sp

    script = (
        "import sys; sys.path.insert(0, 'scripts'); import build_docs_html as b;"
        "print(b.github_base_url())"
    )
    result = sp.run(
        [sys.executable, "-c", script],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    base = result.stdout.strip()
    assert base.startswith("https://github.com/"), f"no GitHub base resolved: {base!r}"
    assert "/blob/" in base, f"base has no blob ref: {base!r}"

    # A real source link must be absolute, not relative into docs_html/.
    html = read("docs/concepts/attention-sinks.html")
    assert 'href="../../models/attention.py"' not in html
    assert "models/attention.py" in html
    for match in re.finditer(r'href="([^"]*models/attention\.py)"', html):
        assert match.group(1).startswith("https://"), (
            f"source link left relative: {match.group(1)}"
        )


def test_manifest_link_rewrite_targets_shipped_twins(built):
    """A .md target without a generated twin must not become a dead .html href."""
    manifest = set(re.findall(
        r'\("([^"]+\.md)", "(?:Core|Concepts|Guides|References)"',
        (ROOT / "scripts" / "build_docs_html.py").read_text(encoding="utf-8"),
    ))
    shipped = {rel.replace(".md", ".html") for rel in manifest}
    shipped.add("index.html")

    for page in built.rglob("*.html"):
        if page.relative_to(built).as_posix() in STATIC_HTML:
            continue
        for match in re.finditer(r'href="([^"#]+\.html)"', page.read_text(encoding="utf-8")):
            href = match.group(1)
            # Absolute URLs point at GitHub blobs, not into docs_html/.
            if href.startswith(("http://", "https://", "//")):
                continue
            target = (page.parent / href).resolve()
            if target.exists():
                continue
            # Only pages this build generates must exist; a static asset may
            # legitimately link outside the site.
            if page.relative_to(built).as_posix() in shipped:
                rel = match.group(1)
                raise AssertionError(f"{page.relative_to(built)} -> dead twin {rel}")


def test_generated_pages_have_balanced_tags(built):
    """Regression: an unclosed <div class="doc-header"> left every doc page
    with unbalanced <main>/<body>/<html>, which silently changed page layout."""
    failures = []
    for page in generated_pages(built):
        checker = _BalanceChecker()
        checker.feed(page.read_text(encoding="utf-8"))
        problems = checker.problems + [
            f"unclosed <{tag}> opened at line {pos[0]}" for tag, pos in checker.stack
        ]
        if problems:
            failures.append(f"{page.relative_to(built)}: {problems[0]}")
    assert not failures, "unbalanced markup:\n" + "\n".join(failures)


# --- accessibility landmarks ---------------------------------------------

def test_skip_link_is_first_tab_stop(built):
    doc = read("README.html")
    assert '<a class="skip-link" href="#articleBody">Skip to content</a>' in doc
    # The target must exist and be the main landmark.
    assert '<main class="main-content" id="articleBody">' in doc
    assert '<a class="skip-link" href="#portalContent">Skip to content</a>' in read("index.html")
    assert 'id="portalContent"' in read("index.html")


def test_landmarks_present(built):
    doc = read("README.html")
    assert '<nav class="sidebar-nav" aria-label="Documentation pages">' in doc
    assert '<nav class="breadcrumb" aria-label="Breadcrumb">' in doc
    assert '<nav class="toc-inner" aria-label="On this page">' in doc
    assert '<footer class="doc-footer">' in doc


def test_active_nav_item_marks_current_page(built):
    doc = read("README.html")
    # Exactly one sidebar link is the current page; it must carry aria-current.
    current = doc.count('class="nav-link active" aria-current="page"')
    assert current == 1


def test_search_input_is_labelled(built):
    doc = read("README.html")
    assert '<label class="sr-only" for="navSearch">' in doc
    assert 'id="navSearchStatus" role="status"' in doc


def test_mobile_toggle_reports_state(built):
    doc = read("README.html")
    assert 'aria-controls="sidebar"' in doc
    assert 'aria-expanded="false"' in doc
    js = (OUT / "assets" / "portal.js").read_text(encoding="utf-8")
    assert "setAttribute('aria-expanded'" in js


# --- design system additions ---------------------------------------------

def test_print_stylesheet_present(built):
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert "@media print" in css
    # Chrome must be dropped and links must expand to their targets.
    print_block = css[css.index("@media print"):]
    assert ".sidebar" in print_block
    assert "page-break-inside" in print_block or "break-inside" in print_block


def test_syntax_tokens_use_house_palette(built):
    """github-dark ships cool blues that clash with the espresso palette;
    local tokens must re-map highlight.js classes onto the design variables."""
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert ".hljs-keyword" in css
    assert ".hljs-string" in css
    assert ".hljs-comment" in css
