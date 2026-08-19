"""Build-output contract tests for the docs_html portal in GPT-OSS-Lite.

Runs the real generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, ASCII hero, mono-only fonts, widget
containers. Markdown sources are never modified by the build.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_docs_html.py"
OUT = ROOT / "docs_html"


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, str(BUILD)], check=True, cwd=ROOT)
    return OUT


def read(rel: str) -> str:
    return (OUT / rel).read_text(encoding="utf-8")


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
