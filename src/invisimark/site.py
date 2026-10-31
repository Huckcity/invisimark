"""The public site's pages: shared layout, per-page titles and descriptions, structured data, robots and sitemap.

Pages are fragments under ``static/site/pages`` wrapped in ``static/site/layout.html``. Everything is rendered once when the
service starts (the content only depends on configuration), then served from memory with a gzip copy and an ETag.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

STATIC = Path(__file__).parent / "static" / "site"
PUBLISHED = "2026-10-02"


@dataclass(frozen=True)
class Page:
    path: str
    file: str
    title: str
    description: str
    kind: str = "website"            # website | article
    priority: float = 0.5
    script: str = ""                 # the tool's script, if the page has one
    crumbs: tuple[tuple[str, str], ...] = ()


PAGES: tuple[Page, ...] = (
    Page("/", "index.html", "Invisimark: free invisible watermark to trace who leaked it",
         "Give every recipient their own invisibly marked copy of a photo or video. When a copy leaks, upload it and find out whose it was. Free, no account.",
         priority=1.0, script="app.js"),
    Page("/check", "check.html", "Check if an image or video is marked as AI-generated",
         "Upload an image or video to look for the Invisimark AI watermark and any signed C2PA metadata. Free, no account, and your file is deleted straight after.",
         priority=0.9, script="check.js"),
    Page("/eu-ai-act-marking", "eu-ai-act-marking.html", "Mark AI images and video for EU AI Act Article 50",
         "An open-source tool for AI image and video apps: an imperceptible watermark, signed C2PA metadata and a free public checker. What it covers, and what it does not.",
         priority=0.9),
    Page("/about", "about.html", "How it works, and where it stops: Invisimark",
         "How per-recipient invisible watermarks work, what they survive (recompression, crops, screen recordings), what defeats them, and what a result means.",
         priority=0.7),
    Page("/guides", "guides.html", "Guides: tracing leaks with invisible watermarks",
         "Practical, honest guides to finding out who leaked a photo or video, watermarking course videos per student, and invisible vs visible watermarks.",
         priority=0.6),
    Page("/guides/find-who-leaked-your-video", "guides/find-who-leaked-your-video.html", "How to find out who leaked your video",
         "Why identical copies make leaks untraceable, how a different invisible mark in every copy fixes it, how to read a result, and how much leaked footage you need.",
         kind="article", priority=0.8, crumbs=(("Guides", "/guides"),)),
    Page("/guides/watermark-course-videos", "guides/watermark-course-videos.html", "Watermarking paid course videos, per student",
         "How to trace a pirated course lesson to the student it came from, without re-rendering the video for every student, and what to expect on screen recordings.",
         kind="article", priority=0.8, crumbs=(("Guides", "/guides"),)),
    Page("/guides/invisible-vs-visible-watermarks", "guides/invisible-vs-visible-watermarks.html", "Invisible vs visible watermarks: which do you need?",
         "A side-by-side comparison of visible and invisible per-recipient watermarks: what each does, how easily it is removed, and when to use both.",
         kind="article", priority=0.7, crumbs=(("Guides", "/guides"),)),
    Page("/privacy", "privacy.html", "Privacy: Invisimark", "What Invisimark does with the files you upload, the keys and names in your case file, and advertising.",
         priority=0.2),
    Page("/terms", "terms.html", "Terms: Invisimark", "What you may use Invisimark for, what it is not, and its limits.", priority=0.2),
)
BY_PATH = {p.path: p for p in PAGES}

# (question, answer as HTML). The same list feeds the page and its FAQPage structured data.
FAQ: tuple[tuple[str, str], ...] = (
    ("Will people notice the mark?",
     "No. It is a faint pattern spread through the texture of the picture, at about the level of ordinary recompression noise, and its "
     "strength is chosen for each file. Flat or dark pictures hide less, and the tool tells you when yours is one of those."),
    ("Can the mark be removed?",
     "Not by accident. It survives recompression, resizing, cropping, mirroring, colour changes, blur and screen recordings (measured on "
     "textured footage). A determined person can damage it with strong noise plus blur, or by averaging copies from several recipients, "
     "so treat it as deterrence plus evidence, not a lock."),
    ("Is a match proof of who leaked it?",
     "No. It is evidence. Someone may have passed their copy on, or their account may have been compromised. Check the result against how "
     "and when you sent each copy. When the evidence is weak the tool says so and does not name anyone."),
    ("How long does a leaked video need to be?",
     "Videos are cut into short segments, and naming one person with confidence takes roughly 25 to 30 clear ones. For a long original "
     "that is about two minutes of leaked footage, for a short clip about half a minute. A shorter leak gives a lead or nothing."),
    ("Does it work on photos?",
     "Yes. Mark the full-size original, not a thumbnail: the mark repeats every 128 pixels, so small pictures and heavy crops carry less."),
    ("What are the limits?",
     "Files up to {{max_mb}} MB, videos up to about {{max_minutes}} minutes of 1080p, and up to {{max_recipients}} recipients per run. "
     "It is free and needs no account."),
    ("What happens to my files and key?",
     "Your key goes into the case file you download and this site does not keep it. The privacy page says exactly what is kept and for how "
     "long: <a href=\"/privacy\">read it here</a>."),
)


def _plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def faq_html() -> str:
    return "\n".join(f"<details><summary>{html.escape(q)}</summary><p>{a}</p></details>" for q, a in FAQ)


@dataclass
class Rendered:
    body: bytes
    gz: bytes
    etag: str
    status: int = 200


def pack(text: str, status: int = 200) -> Rendered:
    body = text.encode("utf-8")
    return Rendered(body, gzip.compress(body, 6, mtime=0), '"' + hashlib.sha256(body).hexdigest()[:20] + '"', status)


def _ld(data: dict) -> str:
    return '<script type="application/ld+json">' + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + "</script>"


class Site:
    """Renders every page once for one configuration."""

    def __init__(self, tokens: dict[str, str], base: str = "", ad_top: str = "", ad_bottom: str = ""):
        self.tokens = tokens
        self.base = base.rstrip("/")
        self.ads = {"ad_top": ad_top, "ad_bottom": ad_bottom}
        self.layout = (STATIC / "layout.html").read_text()
        self.pages = {p.path: pack(self._render(p)) for p in PAGES}
        self.not_found = pack(self._render_404(), 404)

    # -- helpers --------------------------------------------------------------------------

    def _fill(self, text: str, extra: dict[str, str] | None = None) -> str:
        for key, value in {**self.tokens, **(extra or {})}.items():
            text = text.replace("{{" + key + "}}", value)
        return text

    def _jsonld(self, page: Page, description: str) -> str:
        url = self.base + page.path
        if page.path == "/":
            graph = [
                {"@type": "WebSite", "name": "Invisimark", "url": self.base + "/"},
                {"@type": "SoftwareApplication", "name": "Invisimark", "url": self.base + "/", "description": description,
                 "applicationCategory": "MultimediaApplication", "operatingSystem": "Any (web browser)",
                 "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}},
                {"@type": "FAQPage", "mainEntity": [
                    {"@type": "Question", "name": q,
                     "acceptedAnswer": {"@type": "Answer", "text": _plain(self._fill(a))}} for q, a in FAQ]},
            ]
            return _ld({"@context": "https://schema.org", "@graph": graph})
        crumbs = [("Invisimark", "/")] + list(page.crumbs) + [(page.title, page.path)]
        breadcrumb = {"@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": i + 1, "name": name, "item": self.base + target} for i, (name, target) in enumerate(crumbs)]}
        graph: list[dict] = [breadcrumb]
        if page.kind == "article":
            graph.append({"@type": "Article", "headline": page.title, "description": description, "datePublished": PUBLISHED,
                          "dateModified": PUBLISHED, "mainEntityOfPage": url,
                          "author": {"@type": "Organization", "name": "Invisimark"},
                          "publisher": {"@type": "Organization", "name": "Invisimark"}})
        return _ld({"@context": "https://schema.org", "@graph": graph})

    def _nav(self, path: str) -> dict[str, str]:
        current = ' aria-current="page"'
        return {"nav_tool": current if path == "/" else "", "nav_about": current if path == "/about" else "",
                "nav_check": current if path == "/check" else "", "nav_ai": current if path == "/eu-ai-act-marking" else "",
                "nav_guides": current if path.startswith("/guides") else ""}

    def _wrap(self, page: Page | None, content: str, *, title: str, description: str, path: str, jsonld: str = "",
              robots: str = "", og_type: str = "website", scripts: str = "") -> str:
        canonical = self.base + path if self.base else ""
        extra = {
            "title": html.escape(title), "description": html.escape(description), "og_type": og_type, "base": self.base,
            "canonical": canonical or path, "canonical_tag": f'<link rel="canonical" href="{canonical}">' if canonical else "",
            "robots_meta": robots, "jsonld": jsonld, "content": content, "scripts": scripts, **self.ads, **self._nav(path),
        }
        # the layout may carry the tokens, and so may the fragment: fill the fragment first, then the layout
        return self._fill(self.layout, extra)

    def _render(self, page: Page) -> str:
        fragment = (STATIC / "pages" / page.file).read_text()
        content = self._fill(fragment, {"faq": self._fill(faq_html())})
        scripts = f'<script src="/static/{page.script}"></script>' if page.script else ""
        return self._wrap(page, content, title=page.title, description=page.description, path=page.path,
                          jsonld=self._jsonld(page, page.description), og_type="article" if page.kind == "article" else "website",
                          scripts=scripts)

    def _render_404(self) -> str:
        content = ('<article class="prose"><h1>Page not found</h1><p>That page does not exist. '
                   '<a href="/">Go to the tool</a> or read the <a href="/guides">guides</a>.</p></article>')
        return self._wrap(None, content, title="Page not found: Invisimark", description="That page does not exist.", path="/404",
                          robots='<meta name="robots" content="noindex">')

    # -- crawlers -------------------------------------------------------------------------

    def robots(self, base: str = "") -> str:
        base = (self.base or base).rstrip("/")
        return f"User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: {base}/sitemap.xml\n"

    def sitemap(self, base: str = "") -> str:
        base = (self.base or base).rstrip("/")
        today = time.strftime("%Y-%m-%d")
        rows = "".join(
            f"  <url><loc>{html.escape(base + p.path)}</loc><lastmod>{PUBLISHED if p.kind == 'article' else today}</lastmod>"
            f"<priority>{p.priority:.1f}</priority></url>\n" for p in PAGES)
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + rows + "</urlset>\n")
