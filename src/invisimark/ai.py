"""Marking AI-generated images and video, and checking for those marks.

Two layers, following Measure 1.1 of the EU Code of Practice on Transparency of AI-Generated Content:

* an imperceptible watermark derived from a key published here, so anyone can check for it without asking the provider;
* a signed, time-stamped C2PA manifest that declares the file AI-generated and names who made it.

The watermark is public on purpose, which has a cost: anyone can read it, copy it onto another file, or subtract it almost
perfectly (bench/results/ai-adversarial.md). A provider can add a second tile from a secret key, which survives the public tile
being subtracted, so it can still recognise its own output. That tile can be copied along with the public one, so it does not
prove origin. Only the C2PA signature does, for as long as the metadata survives. See docs/spec.md.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, calibrate, detect, embed, keys, media
from .tile import DEFAULT, id_tile, pattern

AI_KEY = hashlib.sha256(b"invisimark.ai-marking.v1").digest()
AI_LABEL = "ai-generated"
PROVIDER_LABEL = "provider"
SCHEME = "invisimark-public-v1"
DETECTOR = "https://invisimark.app"

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
GENERATED = IPTC + "trainedAlgorithmicMedia"
MANIPULATED = IPTC + "compositeWithTrainedAlgorithmicMedia"
AI_SOURCE_TYPES = {GENERATED, MANIPULATED, IPTC + "compositeSynthetic", IPTC + "algorithmicMedia"}
DEFAULT_TIMESTAMPER = "http://timestamp.digicert.com"
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".tif": "image/tiff",
        ".tiff": "image/tiff", ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime"}
VIDEO_OUTPUTS = {".mp4", ".m4v", ".mov"}
DEFAULT_RMS = 1.75              # chosen in bench/ai.py: see bench/results/ai.md
PARENT = "source"
# The search aligns on the whole public pattern, id tile included, so the id score at that alignment is not independent evidence
# and detect's own "gate plus a named candidate" rule does not apply. The alignment score alone decides. See bench/results/ai-null.txt.
PUBLIC_SURE = 8.0
PROVIDER_ALPHA = 2.0            # weight of the private tile beside the public pair, before the overall strength is applied


class MarkError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# marking


@dataclass
class Signing:
    """A C2PA signing identity: a PEM certificate chain (leaf first) and its private key."""

    cert: Path | str
    key: Path | str
    timestamper: str | None = DEFAULT_TIMESTAMPER
    alg: str = "ES256"


@dataclass
class Marked:
    output: str
    sha256: str
    strength: float
    robust: bool | None         # None when the strength was fixed, not measured on this file
    signed: bool
    provider_tile: bool
    calibration: str = ""


def ai_tile(provider_key: bytes | None = None) -> np.ndarray:
    tile = pattern(AI_KEY, AI_LABEL, DEFAULT)
    if provider_key:
        tile = tile + PROVIDER_ALPHA * id_tile(provider_key, PROVIDER_LABEL, DEFAULT)
    return tile


def _manifest(path: Path, *, generated: bool, provider: str | None, system: str | None, model: str | None) -> dict:
    origin = {"provider": provider, "system": system, "model": model}
    agent = {"softwareAgent": {"name": system}} if system else {}
    if generated:
        actions = [{"action": "c2pa.created", "digitalSourceType": GENERATED, **agent}]
    else:   # C2PA requires an edit to open its source first: _sign attaches that source as the ingredient named PARENT
        actions = [{"action": "c2pa.opened", "parameters": {"ingredientIds": [PARENT]}},
                   {"action": "c2pa.edited", "digitalSourceType": MANIPULATED, **agent}]
    return {
        "claim_generator_info": [{"name": "invisimark", "version": __version__}],
        "format": MIME[path.suffix.lower()],
        "title": path.name,
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": actions}},
            {"label": "app.invisimark.marking", "data": {
                "watermark": SCHEME, "detector": DETECTOR, **{k: v for k, v in origin.items() if v},
            }},
        ],
    }


def _sign(src: Path, dst: Path, signing: Signing, manifest: dict, parent: Path | None) -> None:
    import c2pa

    with c2pa.Signer.from_info(c2pa.C2paSignerInfo(
        signing.alg.encode(),
        Path(signing.cert).read_bytes(), Path(signing.key).read_bytes(),
        signing.timestamper.encode() if signing.timestamper else None,
    )) as signer, c2pa.Builder(manifest) as builder:
        if parent is not None:
            with open(parent, "rb") as fh:
                builder.add_ingredient({"title": parent.name, "relationship": "parentOf", "label": PARENT}, MIME[parent.suffix.lower()], fh)
        builder.sign_file(src, dst, signer)


def _has_manifest(path: Path) -> bool:
    return read_manifest(path).get("present", False) if path.suffix.lower() in MIME else False


def mark(
    src: Path | str, dst: Path | str, *, signing: Signing | None = None, generated: bool = True, provider: str | None = None,
    system: str | None = None, model: str | None = None, strength: float | str = DEFAULT_RMS,
    provider_key: bytes | None = None, crf: int = 18, log: Path | str | None = None,
) -> Marked:
    """Watermark `src` and, with `signing`, attach a signed manifest. Signing comes last, so its hash covers the final bytes."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise MarkError(f"{src}: no such file")
    suffix = dst.suffix.lower()
    is_image = media.is_image(src)
    if signing and suffix not in MIME:
        raise MarkError(f"{dst.name}: signed output must be one of {', '.join(sorted(MIME))}")
    if not is_image and suffix not in VIDEO_OUTPUTS:
        raise MarkError(f"{dst.name}: video output must be .mp4, .m4v or .mov")
    if is_image and not media.is_image(dst):
        raise MarkError(f"{dst.name}: image output must be an image format")

    robust, note = None, ""
    if strength == "auto":
        cal = calibrate.calibrate(src, AI_KEY, label=AI_LABEL)
        rms, robust, note = cal.rms, cal.passed, cal.summary()
    else:
        rms = float(strength)
    tile = ai_tile(provider_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".invisimark-") as tmp:
        work = Path(tmp) / f"marked{suffix}"
        if is_image:
            embed.embed_image(src, work, AI_KEY, AI_LABEL, rms=rms, tile=tile)
        else:
            embed.embed_video(src, work, AI_KEY, AI_LABEL, rms=rms, crf=crf, tile=tile)
        if signing:
            # an edit always names its source; a generated file names one only to keep existing provenance (Measure 1.2)
            parent = src if not generated or _has_manifest(src) else None
            if parent is not None and src.suffix.lower() not in MIME:
                raise MarkError(f"{src.name}: signing an edit needs a source in one of {', '.join(sorted(MIME))}")
            signed = Path(tmp) / f"signed{suffix}"
            _sign(work, signed, signing, _manifest(dst, generated=generated, provider=provider, system=system, model=model), parent)
            work = signed
        shutil.move(work, dst)
    digest = keys.sha256_of(dst)
    if log:
        keys.record_issue(Path(log), kind="ai-mark", output_sha256=digest, watermark=SCHEME, signed=bool(signing), strength=rms)
    return Marked(str(dst), digest, rms, robust, bool(signing), bool(provider_key), note)


# ---------------------------------------------------------------------------------------------
# development certificates


def dev_certificate(out_dir: Path | str, name: str = "Invisimark development certificate") -> Signing:
    """A throwaway CA and leaf that C2PA accepts structurally. Anything signed with it validates as untrusted, which is correct."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC)
    ca_key, leaf_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())

    def subject(cn: str) -> x509.Name:
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn), x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NOT TRUSTED")])

    usage = {"content_commitment": False, "key_encipherment": False, "data_encipherment": False, "key_agreement": False,
             "encipher_only": False, "decipher_only": False}
    ca = (x509.CertificateBuilder().subject_name(subject(f"{name} CA")).issuer_name(subject(f"{name} CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=825))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, key_cert_sign=True, crl_sign=True, **usage), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(subject(name)).issuer_name(ca.subject)
            .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=397))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False, **usage), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    cert_path, key_path = out / "dev-chain.pem", out / "dev-key.pem"
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM))
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return Signing(cert_path, key_path)


# ---------------------------------------------------------------------------------------------
# checking


def _source_types(manifest: dict) -> list[str]:
    found = []
    for assertion in manifest.get("assertions", []):
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions", []):
                if action.get("digitalSourceType"):
                    found.append(action["digitalSourceType"])
    return found


def read_manifest(path: Path | str, trust_anchors: str | None = None) -> dict:
    """What the file's C2PA manifest says. A file without one, or one we cannot parse, is reported as such and never raises."""
    import c2pa

    path = Path(path)
    if path.suffix.lower() not in MIME:
        return {"present": False, "note": "this file type is not checked for metadata"}
    try:
        # A file can name a URL for its manifest, and by default the reader fetches it. Files here are untrusted, so it never does.
        settings = {"verify": {"remote_manifest_fetch": False}}
        if trust_anchors:
            settings = {"verify": {"remote_manifest_fetch": False, "verify_trust": True}, "trust": {"trust_anchors": trust_anchors}}
        with c2pa.Context(c2pa.Settings.from_dict(settings)) as context, c2pa.Reader(path, context=context) as reader:
            store = json.loads(reader.json())
            state = reader.get_validation_state()
    except c2pa.C2paError as exc:
        missing = type(exc).__name__ in {"ManifestNotFound", "NotFound"} or "ManifestNotFound" in str(exc) or "JumbfNotFound" in str(exc)
        return {"present": False} if missing else {"present": False, "unreadable": True, "note": "a manifest could not be read from this file"}
    active = store.get("manifests", {}).get(store.get("active_manifest"), {})
    if not active:
        return {"present": False}
    sig = active.get("signature_info") or {}
    types = _source_types(active)
    origin = next((a.get("data") or {} for a in active.get("assertions", []) if a.get("label") == "app.invisimark.marking"), {})
    return {
        "present": True,
        "state": state,                         # Trusted (chains to a loaded list) | Valid | Invalid
        "intact": state in {"Trusted", "Valid"},
        "trusted": state == "Trusted",
        "ai_declared": any(t in AI_SOURCE_TYPES for t in types),
        "source_types": types,
        "signer": sig.get("common_name") or sig.get("issuer"),
        "issuer": sig.get("issuer"),
        "signed_at": sig.get("time"),
        "provider": origin.get("provider"),
        "system": origin.get("system"),
        "model": origin.get("model"),
    }


@dataclass
class Check:
    sha256: str
    watermark: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    ai_marked: bool = False
    summary: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return asdict(self)


def find_mark(frames: list[np.ndarray], provider_key: bytes | None = None) -> detect.Report:
    """Look for the public mark, or a provider's private tile. If the plain search misses it, try again allowing for a tilt or a
    change of aspect ratio."""
    tile = id_tile(provider_key, PROVIDER_LABEL, DEFAULT) if provider_key else ai_tile()
    key, label = (provider_key, PROVIDER_LABEL) if provider_key else (AI_KEY, AI_LABEL)
    rep = detect.detect_frames(frames, key, [label], search=tile)
    if rep.sync_z >= PUBLIC_SURE:
        return rep
    _, geometry = detect.search_geometry(frames, tile)
    again = detect.detect_frames(detect.rectify(frames, geometry), key, [label], scale_hint=1.0, search=tile)
    return again if again.sync_z > rep.sync_z else rep


def read_watermark(path: Path | str, provider_key: bytes | None = None, frames: int = 32) -> dict:
    loaded = detect.load_frames(path, frames)
    rep = find_mark(loaded)
    id_z = rep.matches[0].z if rep.matches else 0.0
    present = rep.sync_z >= PUBLIC_SURE
    out = {"scheme": SCHEME, "present": present, "sync_score": round(rep.sync_z, 1), "id_score": round(id_z, 1),
           "scale": round(rep.scale, 3), "reoriented": rep.orientation != 0}
    if provider_key:
        private = find_mark(loaded, provider_key).sync_z
        out["provider_score"] = round(private, 1)
        out["provider_verified"] = private >= PUBLIC_SURE
    return out


def _describe(wm: dict, meta: dict) -> tuple[bool, list[str]]:
    lines = []
    if wm.get("present"):
        lines.append("Watermark: this file carries the public Invisimark AI-marking watermark. It says the file was marked as AI-generated "
                     "or AI-manipulated. It does not say by whom, and it can be copied onto another file.")
    else:
        lines.append("Watermark: the public Invisimark watermark was not found. Other providers use other watermarks, which this check cannot read.")
    if wm.get("provider_verified"):
        lines.append("Provider key: the private tile for the key you supplied is present. It can be copied along with the public mark, so it does not prove the file is yours.")
    if not meta.get("present"):
        if meta.get("unreadable"):
            lines.append("Metadata: this file holds provenance data that could not be read.")
        else:
            lines.append("Metadata: no C2PA manifest found. Most platforms strip metadata when files are shared, so this says little on its own.")
    else:
        who = meta.get("signer") or "an unnamed signer"
        if not meta.get("intact"):
            lines.append(f"Metadata: a C2PA manifest from {who} is present, but its signature does not validate, so what it says cannot be relied on. "
                         "The file may have been changed after signing.")
        else:
            trust = "and chains to the C2PA trust list" if meta.get("trusted") else "but the certificate is not on the trust list used here, so the name is the signer's own claim"
            claim = "It declares the file AI-generated or AI-manipulated." if meta.get("ai_declared") else "It does not declare the file AI-generated."
            lines.append(f"Metadata: a C2PA manifest signed by {who}. The signature is intact, {trust}. {claim}")
    ai_marked = bool(wm.get("present") or (meta.get("intact") and meta.get("ai_declared")))
    if not ai_marked:
        lines.append("No AI marking found. That does not mean a person made this file: marks can be missing, removed or in a format this check cannot read.")
    return ai_marked, lines


def check(path: Path | str, provider_key: bytes | None = None, trust_anchors: str | None = None, frames: int = 32) -> Check:
    path = Path(path)
    wm = read_watermark(path, provider_key, frames)
    meta = read_manifest(path, trust_anchors)
    ai_marked, lines = _describe(wm, meta)
    return Check(keys.sha256_of(path), wm, meta, ai_marked, lines)


# ---------------------------------------------------------------------------------------------
# signed detection results (Sub-measure 2.1.2: a hash of the content, an identifier of the detector, a timestamp)


def canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def report_keypair() -> tuple[str, str]:
    """A new Ed25519 pair for signing detection results, as (private hex, public hex)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return raw.hex(), private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def public_key_of(private_hex: str) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def signed_report(result: Check, private_hex: str, detector: str = DETECTOR) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    body = {
        "type": "invisimark-detection-result", "version": 1, "detector": detector, "software": f"invisimark {__version__}",
        "time": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "content_sha256": result.sha256,
        "ai_marked": result.ai_marked, "watermark": result.watermark, "metadata": result.metadata, "summary": result.summary,
    }
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex)).sign(canonical(body))
    return {"result": body, "alg": "Ed25519", "public_key": public_key_of(private_hex), "signature": base64.b64encode(signature).decode()}


def verify_report(report: dict, public_hex: str | None = None) -> bool:
    """True when the signature matches the result, under `public_hex` if given. Without it only integrity is shown, not who signed."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        key = public_hex or report["public_key"]
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(base64.b64decode(report["signature"]), canonical(report["result"]))
        return True
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return False
