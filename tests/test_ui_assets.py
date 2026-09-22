"""Static consistency checks for the chat client.

The UI has no build step and no test runner of its own, so the cheap failure
mode is a silent mismatch between `app.js` and `index.html` (a renamed id, a
dropped element) or between the client and the SSE contract the API emits.
These tests catch exactly that, with no browser and no npm.
"""

from __future__ import annotations

import json
import re

import pytest
from src.main import STATIC_DIR
from src.models import DoneEvent, ErrorEvent, SourceReference, SourcesEvent, TokenEvent

INDEX = STATIC_DIR / "index.html"
APP_JS = STATIC_DIR / "app.js"
STYLES = STATIC_DIR / "styles.css"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    return APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return STYLES.read_text(encoding="utf-8")


def test_assets_exist() -> None:
    assert INDEX.is_file() and APP_JS.is_file() and STYLES.is_file()


def test_every_element_the_script_looks_up_exists_in_the_page(html: str, script: str) -> None:
    referenced = set(re.findall(r'getElementById\("([^"]+)"\)', script))
    declared = set(re.findall(r'id="([^"]+)"', html))

    assert referenced, "the client must bind to the page"
    missing = referenced - declared
    assert not missing, f"app.js binds to ids that index.html does not define: {sorted(missing)}"


def test_every_class_the_script_creates_is_styled(script: str, css: str) -> None:
    # `element(tag, className, …)` is the only place the client creates classes.
    created = {
        token
        for match in re.findall(r'element\("[a-z]+", "([^"]+)"', script)
        for token in match.split()
    }
    styled = set(re.findall(r"\.([A-Za-z][\w-]*)", css))

    assert created, "the client must attach styles"
    unstyled = created - styled
    assert not unstyled, f"app.js uses unstyled classes: {sorted(unstyled)}"


def test_client_consumes_every_sse_event_the_api_emits(script: str) -> None:
    handled = set(re.findall(r'frame\.event === "([a-z]+)"', script))

    emitted = {
        SourcesEvent.model_fields["type"].default.value,
        TokenEvent.model_fields["type"].default.value,
        DoneEvent.model_fields["type"].default.value,
        ErrorEvent.model_fields["type"].default.value,
    }
    assert handled == emitted, "the UI and the stream contract have drifted apart"


def test_client_reads_the_fields_the_payloads_actually_carry(script: str) -> None:
    for field in ("retrieval_ms", "candidates_considered", "trace_id"):
        assert field in SourcesEvent.model_fields
        assert field in script
    for field in ("source", "page", "rerank_score", "dense_score", "preview"):
        assert field in SourceReference.model_fields
        assert field in script
    assert "usage" in DoneEvent.model_fields
    assert "input_tokens" in script and "output_tokens" in script


def test_request_body_matches_the_endpoint_contract(script: str) -> None:
    from src.models import QueryRequest

    body = re.search(r"JSON\.stringify\(\{(.*?)\}\)", script, re.DOTALL)
    assert body is not None
    sent = set(re.findall(r"(\w+):", body.group(1)))

    assert sent <= set(QueryRequest.model_fields), "the UI sends fields the API rejects"
    assert "query" in sent

    # The client must enforce the same bounds the API validates against, so a
    # bad query is rejected before it costs a round trip.
    assert "const MIN_QUERY = 3" in script
    assert "const MAX_QUERY = 500" in script
    assert any(getattr(item, "le", None) == 10 for item in QueryRequest.model_fields["k"].metadata)
    for option in re.findall(r'<option value="(\d+)"', INDEX.read_text(encoding="utf-8")):
        assert 1 <= int(option) <= 10


def test_markdown_is_rendered_from_escaped_text(script: str) -> None:
    """LLM output derives from untrusted PDFs: HTML must be escaped first."""
    assert "escapeHtml" in script
    renderer = script[script.index("const renderMarkdown") : script.index("const formatMs")]
    assert "escapeHtml(text)" in renderer
    assert "marked.parse(escaped" in renderer


def test_cdn_script_is_pinned_and_integrity_checked(html: str) -> None:
    tag = re.search(r"<script[^>]*marked[^>]*></script>", html, re.DOTALL)
    assert tag is not None, "marked.js must be loaded from a pinned CDN URL"
    assert re.search(r"marked@\d+\.\d+\.\d+/marked\.min\.js", tag.group(0))
    assert 'integrity="sha384-' in tag.group(0)
    assert 'crossorigin="anonymous"' in tag.group(0)


def test_page_has_no_inline_script_or_style_blocks(html: str) -> None:
    """Keep CSP-friendly: behaviour lives in app.js, styling in styles.css."""
    assert "<script>" not in html
    assert "<style>" not in html
    assert "/static/app.js" in html and "/static/styles.css" in html


def test_page_is_responsive_and_accessible(html: str, css: str) -> None:
    assert 'name="viewport"' in html
    assert 'aria-live="polite"' in html
    assert 'lang="en"' in html
    assert "@media (max-width:" in css
    assert "prefers-reduced-motion" in css


def test_suggestion_chips_are_valid_queries(html: str) -> None:
    from src.models import QueryRequest

    for question in re.findall(r'data-q="([^"]+)"', html):
        assert QueryRequest(query=question).query == question


def test_sse_frames_from_the_api_parse_with_the_client_grammar() -> None:
    """Mirror of `parseFrame` in app.js, exercised against a real payload."""
    event = TokenEvent(text="hello")
    frame = f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"

    name = "message"
    data: list[str] = []
    for line in frame.strip().split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())

    assert name == "token"
    assert json.loads("\n".join(data))["text"] == "hello"
