"""
multimodal.py — Vision / Image Content Builder
===============================================
Handles image inputs across all providers.

Canonical internal format
-------------------------
  OpenAI ``image_url`` content blocks:
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..." | "https://..."}}

  This is what kernel.py writes into messages when images are attached.
  tool_bridge.normalise_messages() then translates to the target provider's
  native format before the request is sent.

Provider translation table
--------------------------
  openai / groq / mistral / together / gemini / ollama / openrouter / lmstudio / deepseek
      → pass-through (already OpenAI-compatible image_url blocks)
  anthropic
      → {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}
        or {"type":"image","source":{"type":"url","url":"..."}}
  hf_local / llamacpp
      → strip image blocks, prepend "[image attached]" note to text

Public API
----------
  build_content_with_images(text, images, provider) → list[dict]
  normalise_image_blocks_for_provider(content, provider) → list[dict]
  provider_supports_vision(provider) → bool
  openai_image_block(url_or_data_uri) → dict
  load_image_to_data_uri(path_or_url) → (data_uri, media_type)
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("essence.multimodal")

# ---------------------------------------------------------------------------
# Provider capability table
# ---------------------------------------------------------------------------

_VISION_PROVIDERS = frozenset({
    "openai", "anthropic", "groq", "mistral", "together",
    "gemini", "ollama", "openrouter", "lmstudio", "deepseek",
})

_ANTHROPIC_PROVIDERS = frozenset({"anthropic"})
_NO_VISION_PROVIDERS  = frozenset({"hf_local", "llamacpp"})


def provider_supports_vision(provider: str) -> bool:
    """Return True if the provider accepts image content blocks."""
    return provider in _VISION_PROVIDERS


# ---------------------------------------------------------------------------
# Media type helpers
# ---------------------------------------------------------------------------

_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff",  "image/jpeg"),
    (b"\x89PNG\r\n",   "image/png"),
    (b"GIF87a",        "image/gif"),
    (b"GIF89a",        "image/gif"),
    (b"RIFF",          "image/webp"),   # followed by WEBP at offset 8
    (b"IDAT",          "image/png"),
    (b"<svg",          "image/svg+xml"),
]

def _detect_media_type(data: bytes) -> str:
    """Detect MIME type from the first few bytes of image data."""
    for magic, mime in _MAGIC:
        if data[:len(magic)] == magic:
            # WebP needs secondary check
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return "image/jpeg"   # safe default


def _data_uri_media_type(data_uri: str) -> str:
    """Extract media type from a data: URI."""
    m = re.match(r"data:([^;]+);base64,", data_uri)
    return m.group(1) if m else "image/jpeg"


def _data_uri_to_b64(data_uri: str) -> str:
    """Return the raw base64 payload from a data: URI."""
    if ";base64," in data_uri:
        return data_uri.split(";base64,", 1)[1]
    return data_uri


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_to_data_uri(source: str) -> tuple[str, str]:
    """
    Load an image from a local path, http(s) URL, or existing data: URI.

    Returns (data_uri, media_type).

    For http(s) URLs the function does NOT fetch the content — instead
    it returns the URL itself as the ``data_uri`` value (callers must
    handle URL-native providers separately).  Use ``is_url()`` to check.
    """
    if source.startswith("data:"):
        mt = _data_uri_media_type(source)
        return source, mt

    if re.match(r"https?://", source):
        # Return as bare URL — caller decides whether to fetch
        return source, "image/url"

    # Local file path
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    data = path.read_bytes()
    mt   = _detect_media_type(data)

    # Fallback to mimetypes module
    if mt == "image/jpeg" and path.suffix.lower() in (".png", ".gif", ".webp"):
        mt = mimetypes.guess_type(str(path))[0] or mt

    b64      = base64.standard_b64encode(data).decode()
    data_uri = f"data:{mt};base64,{b64}"
    return data_uri, mt


def is_url(source: str) -> bool:
    return re.match(r"https?://", source) is not None


# ---------------------------------------------------------------------------
# OpenAI canonical format
# ---------------------------------------------------------------------------

def openai_image_block(source: str) -> dict:
    """
    Build an OpenAI-canonical image content block.

    ``source`` may be:
      • https://...          → image_url with url
      • data:image/...       → image_url with data URI
      • /local/path.jpg      → loaded, converted to data URI
    """
    if not source.startswith("data:") and not is_url(source):
        source, _ = load_image_to_data_uri(source)

    return {"type": "image_url", "image_url": {"url": source}}


# ---------------------------------------------------------------------------
# Anthropic format
# ---------------------------------------------------------------------------

def _to_anthropic_image_block(openai_block: dict) -> dict:
    """
    Translate one OpenAI image_url block to Anthropic's image block.

    OpenAI data URI  → Anthropic base64 source
    OpenAI https URL → Anthropic url source
    """
    url: str = openai_block.get("image_url", {}).get("url", "")

    if url.startswith("data:"):
        mt  = _data_uri_media_type(url)
        b64 = _data_uri_to_b64(url)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": b64},
        }

    # Plain URL
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


# ---------------------------------------------------------------------------
# Content-list normalisation
# ---------------------------------------------------------------------------

def normalise_image_blocks_for_provider(
    content: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, Any]]:
    """
    Translate image blocks inside a multimodal content list to the
    target provider's wire format.

    Input  : OpenAI-canonical content list (may contain text + image_url blocks)
    Output : Provider-native content list

    Non-image blocks (type=="text") are passed through unchanged.
    Providers that don't support vision get image blocks stripped and a
    short note appended to the preceding text block.
    """
    if not content or not isinstance(content, list):
        return content

    # Check if there are any image blocks at all
    has_images = any(
        isinstance(b, dict) and b.get("type") == "image_url"
        for b in content
    )
    if not has_images:
        return content

    if provider in _ANTHROPIC_PROVIDERS:
        out: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                out.append(block)
                continue
            if block.get("type") == "image_url":
                out.append(_to_anthropic_image_block(block))
            else:
                out.append(block)
        return out

    if provider in _NO_VISION_PROVIDERS:
        # Strip images, add a text note
        texts    = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        n_images = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image_url")
        note     = f"[{n_images} image(s) attached — not supported by {provider}, stripped]"
        return [{"type": "text", "text": "\n".join(texts) + "\n" + note}]

    # All OpenAI-compatible providers — pass through as-is
    return content


# ---------------------------------------------------------------------------
# High-level builder used by kernel._stream_response()
# ---------------------------------------------------------------------------

def build_content_with_images(
    text:     str,
    images:   list[dict],
    provider: str = "openai",
) -> list[dict] | str:
    """
    Build the ``content`` value for a user message that includes images.

    If ``images`` is empty, returns the plain ``text`` string (no change).
    Otherwise returns a content list: [text_block, *image_blocks].

    Each entry in ``images`` must be a dict with one of:
      {"url":    "https://..."}
      {"path":   "/local/file.jpg"}
      {"base64": "...", "media_type": "image/jpeg"}

    The returned content is in OpenAI canonical format; the tool_bridge
    will translate it to the target provider's format in normalise_messages().
    """
    if not images:
        return text

    blocks: list[dict] = [{"type": "text", "text": text}]

    for img in images:
        try:
            if "base64" in img:
                mt  = img.get("media_type", "image/jpeg")
                uri = f"data:{mt};base64,{img['base64']}"
                blocks.append(openai_image_block(uri))
            elif "path" in img:
                blocks.append(openai_image_block(img["path"]))
            elif "url" in img:
                blocks.append(openai_image_block(img["url"]))
            else:
                log.warning("multimodal: unknown image dict keys: %s", list(img))
        except Exception as exc:
            log.warning("multimodal: failed to load image %s: %s", img, exc)
            blocks.append({"type": "text", "text": f"[image load error: {exc}]"})

    # Immediately translate for the current provider
    return normalise_image_blocks_for_provider(blocks, provider)
