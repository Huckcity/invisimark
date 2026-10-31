import http.server
import json
import threading

import cv2
import numpy as np
import pytest

from invisimark import ai, cli, media

from .conftest import photo_like


@pytest.fixture(scope="module")
def signing(tmp_path_factory):
    made = ai.dev_certificate(tmp_path_factory.mktemp("cert"))
    made.timestamper = None
    return made


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "source.png"
    cv2.imwrite(str(path), photo_like(768, 768))
    return path


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp"])
def test_both_layers_are_found_and_described(tmp_path, signing, source, suffix):
    out = tmp_path / f"out{suffix}"
    done = ai.mark(source, out, signing=signing, provider="Acme", system="AcmeGen", model="m1")
    result = ai.check(out)
    assert done.signed and result.ai_marked
    assert result.watermark["present"]
    assert result.metadata["intact"] and result.metadata["ai_declared"] and not result.metadata["trusted"]
    assert result.metadata["provider"] == "Acme" and result.metadata["signer"] == "Invisimark development certificate"
    assert any(line.startswith("Watermark:") for line in result.summary)
    assert any(line.startswith("Metadata:") and "signer's own claim" in line for line in result.summary)


def test_an_unmarked_file_is_not_called_human_made(source):
    result = ai.check(source)
    assert not result.ai_marked and not result.watermark["present"] and not result.metadata["present"]
    assert any("does not mean a person made this file" in line for line in result.summary)


def test_a_private_key_watermark_is_not_the_public_mark(tmp_path, source):
    from invisimark import embed

    out = tmp_path / "other.png"
    embed.embed_image(source, out, b"someone-elses-key-000000", "ai-generated", rms=ai.DEFAULT_RMS)
    assert not ai.check(out).watermark["present"]


def test_changing_a_signed_file_breaks_the_signature_but_the_watermark_remains(tmp_path, signing, source):
    out = tmp_path / "out.jpg"
    ai.mark(source, out, signing=signing)
    raw = bytearray(out.read_bytes())
    raw[-5000] ^= 1
    out.write_bytes(raw)
    meta = ai.read_manifest(out)
    assert meta["present"] and not meta["intact"]
    assert any("does not validate" in line for line in ai.check(out).summary)


def test_trust_anchors_decide_whether_a_signer_is_trusted(tmp_path, signing, source):
    out = tmp_path / "out.jpg"
    ai.mark(source, out, signing=signing)
    root = signing.cert.read_text().split("-----END CERTIFICATE-----")[1] + "-----END CERTIFICATE-----\n"
    assert ai.read_manifest(out)["state"] == "Valid"
    assert ai.read_manifest(out, root)["state"] == "Trusted"


def test_a_manifest_url_inside_a_file_is_never_fetched(tmp_path, signing, source):
    import c2pa

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    jpg = tmp_path / "in.jpg"
    cv2.imwrite(str(jpg), photo_like(256, 256))
    info = c2pa.C2paSignerInfo(b"ES256", signing.cert.read_bytes(), signing.key.read_bytes(), None)
    with c2pa.Signer.from_info(info) as signer, c2pa.Builder(ai._manifest(jpg, generated=True, provider=None, system=None, model=None)) as builder:
        builder.set_no_embed()
        builder.set_remote_url(f"http://127.0.0.1:{server.server_port}/manifest.c2pa")
        builder.sign_file(jpg, tmp_path / "remote.jpg", signer)
    meta = ai.read_manifest(tmp_path / "remote.jpg")
    server.shutdown()
    assert hits == [] and meta["unreadable"]


def test_an_ai_edit_is_declared_and_keeps_the_earlier_provenance(tmp_path, signing, source):
    import c2pa

    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    ai.mark(source, first, signing=signing, provider="First")
    ai.mark(first, second, signing=signing, generated=False, provider="Second")
    meta = ai.read_manifest(second)
    assert meta["intact"] and meta["ai_declared"] and meta["provider"] == "Second"
    assert ai.MANIPULATED in meta["source_types"]
    with c2pa.Reader(second) as reader:
        store = json.loads(reader.json())
    assert len(store["manifests"]) == 2


def test_an_ai_edit_of_an_unsigned_file_is_valid_too(tmp_path, signing, source):
    out = tmp_path / "out.png"
    ai.mark(source, out, signing=signing, generated=False)
    assert ai.read_manifest(out)["intact"]


def test_a_provider_key_tells_your_output_from_a_copy_of_the_public_mark(tmp_path, source):
    key, other = b"provider-key-0123456789abcdef", b"another-key-0123456789abcdefgh"
    ours, public_only = tmp_path / "ours.png", tmp_path / "public.png"
    ai.mark(source, ours, provider_key=key)
    ai.mark(source, public_only)
    assert ai.read_watermark(ours)["present"]
    assert ai.read_watermark(ours, key)["provider_verified"]
    assert not ai.read_watermark(ours, other)["provider_verified"]
    assert not ai.read_watermark(public_only, key)["provider_verified"]


def test_the_alpha_channel_is_carried_through(tmp_path):
    src, out = tmp_path / "a.png", tmp_path / "b.png"
    alpha = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    cv2.imwrite(str(src), np.dstack([photo_like(512, 512), alpha]))
    ai.mark(src, out)
    assert np.array_equal(cv2.imread(str(out), cv2.IMREAD_UNCHANGED)[..., 3], alpha)


def test_signed_results_detect_any_change_and_name_their_key():
    private, public = ai.report_keypair()
    result = ai.Check("ab" * 32, {"present": True}, {"present": False}, True, ["line"])
    report = ai.signed_report(result, private, "https://detector.example")
    body = report["result"]
    assert body["content_sha256"] == "ab" * 32 and body["detector"] == "https://detector.example" and body["time"].endswith("Z")
    assert ai.verify_report(report, public) and ai.public_key_of(private) == public
    assert not ai.verify_report(report, ai.report_keypair()[1])
    body["ai_marked"] = False
    assert not ai.verify_report(report)


def test_a_playlist_dressed_as_a_video_is_refused(tmp_path):
    fake = tmp_path / "video.mp4"
    fake.write_text("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n")
    with pytest.raises(media.MediaError):
        media.probe(fake)


def test_command_line_marks_checks_and_signs_a_result(tmp_path, source, capsys):
    out, cert, report_key, report = tmp_path / "out.png", tmp_path / "cert", tmp_path / "rk", tmp_path / "result.json"
    assert cli.main(["dev-cert", "--out", str(cert)]) == 0
    assert cli.main(["mark", str(source), str(out), "--cert", str(cert / "dev-chain.pem"), "--signing-key", str(cert / "dev-key.pem"),
                     "--no-timestamp", "--provider", "Acme"]) == 0
    assert cli.main(["report-key", "--out", str(report_key)]) == 0
    assert cli.main(["report-key", "--out", str(report_key)]) == 1
    public = capsys.readouterr().out.split("results:\n")[1].split()[0]
    assert cli.main(["check", str(out), "--report-key", str(report_key), "--report", str(report)]) == 0
    assert cli.main(["verify-report", str(report), "--public-key", public]) == 0
    assert cli.main(["check", str(source)]) == 1
    capsys.readouterr()
    assert cli.main(["mark", str(source), str(tmp_path / "x.png"), "--cert", str(cert / "dev-chain.pem")]) == 1


def test_unmarked_photographs_that_once_fooled_the_id_score_are_not_called_marked(tmp_path):
    from skimage import data

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    for name, side in (("half.png", 384), ("full.png", 768)):
        path = tmp_path / name
        cv2.imwrite(str(path), cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA))
        assert not ai.read_watermark(path)["present"]


@pytest.mark.parametrize("change", ["tilt", "tilt_mirrored", "squeeze"])
def test_the_mark_is_found_after_a_tilt_or_an_aspect_change(tmp_path, source, change):
    marked, out = tmp_path / "marked.png", tmp_path / "changed.jpg"
    ai.mark(source, marked)
    img = cv2.imread(str(marked))
    h, w = img.shape[:2]
    if change.startswith("tilt"):
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)
    if change == "tilt_mirrored":
        img = cv2.flip(img, 1)
    if change == "squeeze":
        img = cv2.resize(img, None, fx=0.75, fy=1.0, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ai.read_watermark(out)["present"]
