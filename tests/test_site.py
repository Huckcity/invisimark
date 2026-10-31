import gzip
import html
import json
import re

import pytest
from starlette.testclient import TestClient

from invisimark import service, site

PUBLIC = "https://invisimark.example"


def make(tmp_path, monkeypatch, public=PUBLIC, contact="contact@invisimark.example"):
    if public:
        monkeypatch.setenv("IM_PUBLIC_URL", public)
    else:
        monkeypatch.delenv("IM_PUBLIC_URL", raising=False)
    monkeypatch.setenv("IM_CONTACT", contact)
    return service.make_app(service.Limits(max_bytes=100 * 1024), tmp_path / "work")


def test_every_page_has_a_unique_title_a_description_a_canonical_and_social_tags(tmp_path, monkeypatch):
    seen = set()
    with TestClient(make(tmp_path, monkeypatch)) as client:
        for page in site.PAGES:
            r = client.get(page.path)
            assert r.status_code == 200 and "{{" not in r.text, page.path
            title = html.unescape(re.search(r"<title>(.*?)</title>", r.text).group(1))
            desc = html.unescape(re.search(r'<meta name="description" content="(.*?)">', r.text).group(1))
            assert 15 <= len(title) <= 62 and title not in seen, (page.path, title)
            assert 60 <= len(desc) <= 165, (page.path, len(desc))
            seen.add(title)
            assert f'<link rel="canonical" href="{PUBLIC}{page.path}">' in r.text
            for tag in (f'<meta property="og:url" content="{PUBLIC}{page.path}">', f'<meta property="og:image" content="{PUBLIC}/static/og.png">',
                        '<meta name="twitter:card" content="summary_large_image">', '<meta property="og:title"'):
                assert tag in r.text, (page.path, tag)
            assert r.text.count("<h1") == 1, page.path
            assert 'lang="en"' in r.text and 'name="viewport"' in r.text


def test_structured_data_parses_and_the_faq_matches_the_page(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        home = client.get("/").text
        blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', home, re.S)
        assert len(blocks) == 1
        graph = json.loads(blocks[0].replace("<\\/", "</"))["@graph"]
        kinds = {g["@type"] for g in graph}
        assert kinds == {"WebSite", "SoftwareApplication", "FAQPage"}
        faq = next(g for g in graph if g["@type"] == "FAQPage")
        faq_section = home[home.index('<section class="block faq"'):]
        assert len(faq["mainEntity"]) == len(site.FAQ) == faq_section.count("<summary>")
        for q in faq["mainEntity"]:
            assert f"<summary>{html.escape(q['name'])}</summary>" in faq_section and "{{" not in q["acceptedAnswer"]["text"]
        assert next(g for g in graph if g["@type"] == "SoftwareApplication")["offers"]["price"] == "0"

        article = client.get("/guides/find-who-leaked-your-video").text
        data = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', article, re.S).group(1))["@graph"]
        assert {g["@type"] for g in data} == {"BreadcrumbList", "Article"}
        crumbs = next(g for g in data if g["@type"] == "BreadcrumbList")["itemListElement"]
        assert [c["position"] for c in crumbs] == [1, 2, 3] and crumbs[0]["item"] == PUBLIC + "/"


def test_robots_and_sitemap(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        robots = client.get("/robots.txt")
        assert robots.headers["content-type"].startswith("text/plain")
        assert "Disallow: /api/" in robots.text and f"Sitemap: {PUBLIC}/sitemap.xml" in robots.text
        sitemap = client.get("/sitemap.xml")
        assert sitemap.headers["content-type"].startswith("application/xml")
        locs = re.findall(r"<loc>(.*?)</loc>", sitemap.text)
        assert sorted(locs) == sorted(PUBLIC + p.path for p in site.PAGES)
        assert all("/api/" not in loc for loc in locs)


def test_without_a_public_url_the_crawler_files_use_the_request_host(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch, public="")) as client:
        assert "Sitemap: http://testserver/sitemap.xml" in client.get("/robots.txt").text
        assert "<loc>http://testserver/about</loc>" in client.get("/sitemap.xml").text
        assert 'rel="canonical"' not in client.get("/").text


def test_every_internal_link_and_asset_resolves(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        targets = set()
        for page in site.PAGES:
            text = client.get(page.path).text
            targets |= {h.split("#")[0] or "/" for h in re.findall(r'(?:href|src)="(/[^"]*)"', text)}
            targets |= set(re.findall(r'content="' + re.escape(PUBLIC) + r'(/[^"]*)"', text))        # og:image, og:url, twitter:image
        assert "/static/og.png" in targets and "/guides" in targets and "/static/favicon.svg" in targets
        for target in sorted(targets):
            assert client.get(target).status_code == 200, target


def test_pages_are_compressed_cached_and_revalidate(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        plain = client.get("/about", headers={"accept-encoding": "identity"})
        packed = client.get("/about", headers={"accept-encoding": "gzip"})
        assert packed.headers["content-encoding"] == "gzip" and "content-encoding" not in plain.headers
        assert int(packed.headers["content-length"]) < int(plain.headers["content-length"]) // 2          # the wire sizes
        assert gzip.decompress(gzip.compress(plain.content)) == packed.content and "Accept-Encoding" in packed.headers["vary"]
        assert "max-age" in packed.headers["cache-control"] and "no-store" not in packed.headers["cache-control"]
        etag = packed.headers["etag"]
        again = client.get("/about", headers={"if-none-match": etag})
        assert again.status_code == 304 and again.content == b""
        png = client.get("/static/og.png", headers={"accept-encoding": "gzip"})
        assert "content-encoding" not in png.headers and png.content[:4] == b"\x89PNG"
        assert client.get("/static/style.css").headers["content-type"].startswith("text/css")


def test_other_hostnames_are_kept_out_of_search_and_the_api_is_never_indexed(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        assert client.get("/", headers={"host": "invisimark.fly.dev"}).headers["x-robots-tag"] == "noindex"
        assert "x-robots-tag" not in client.get("/", headers={"host": "invisimark.example"}).headers
        assert client.get("/api/limits").headers["x-robots-tag"] == "noindex"
        assert client.get("/api/jobs/nothing").headers["x-robots-tag"] == "noindex"


def test_unknown_pages_get_a_real_404_and_the_api_keeps_json(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        missing = client.get("/no-such-page")
        assert missing.status_code == 404 and missing.headers["content-type"].startswith("text/html")
        assert '<meta name="robots" content="noindex">' in missing.text and "Page not found" in missing.text
        assert client.get("/api/jobs/nothing").json()["detail"].startswith("No such job")
        assert client.get("/static/nothing.js").status_code == 404


def test_the_footer_and_policy_text_carry_the_contact_address(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        for page in site.PAGES:
            assert 'mailto:contact@invisimark.example' in client.get(page.path).text, page.path


def test_the_guides_are_substantial_and_link_back_to_the_tool(tmp_path, monkeypatch):
    with TestClient(make(tmp_path, monkeypatch)) as client:
        for path in ("/guides/find-who-leaked-your-video", "/guides/watermark-course-videos", "/guides/invisible-vs-visible-watermarks"):
            text = re.sub(r"<[^>]+>", " ", client.get(path).text)
            assert len(text.split()) > 380, path
            assert 'href="/"' in client.get(path).text
            assert "guarantee" not in text.lower().replace("not a guarantee", "")      # no overclaiming
