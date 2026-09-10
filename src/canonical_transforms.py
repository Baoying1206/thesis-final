"""Frozen, deterministic transforms for the two placeholder-scheme
mechanisms in templates/final_10_condition_v1.json:
mg_encoding_obfuscation ({instruction_b64}) and mg_payload_splitting
({payload_a}/{payload_b}). See FINAL_STUDY_PROTOCOL.md Sec 5.2/5.6 for
the frozen spec these implement verbatim.

Both transforms are pure and deterministic (same input -> same output,
every time, on every machine) and both return their own input/output
SHA-256 hashes alongside the transformed text, so every render is
independently auditable without needing to trust the code that produced
it. Stdlib only (hashlib, base64, re via axis_manifest.normalize_text).
No torch, no tokenizer, no model.
"""

import base64
import hashlib

from axis_manifest import normalize_text, normalized_text_hash


def sha256_hex(b):
    """b may be str (encoded as UTF-8) or bytes."""
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def encode_base64_utf8(instruction_text):
    """Frozen encoding transform (FINAL_STUDY_PROTOCOL.md Sec 5.6):
    UTF-8 encode the RAW instruction text (no prior normalization —
    this transform deliberately does NOT call normalize_text), apply
    standard Base64 (RFC 4648 alphabet, base64.b64encode, no line
    wrapping), decode the result as ASCII to obtain the
    {instruction_b64} substitution string.

    Returns (b64_string, input_sha256, output_sha256):
    - input_sha256 = sha256(utf8_bytes(instruction_text))
    - output_sha256 = sha256(ascii_bytes(b64_string))
    """
    input_bytes = instruction_text.encode("utf-8")
    b64_bytes = base64.b64encode(input_bytes)
    if b"\n" in b64_bytes:
        raise ValueError("base64.b64encode produced an embedded newline -- frozen spec forbids line wrapping")
    b64_string = b64_bytes.decode("ascii")
    return b64_string, sha256_hex(input_bytes), sha256_hex(b64_bytes)


def split_payload_midpoint(instruction_text):
    """Frozen payload-splitting transform (FINAL_STUDY_PROTOCOL.md Sec
    5.6): normalize the instruction via axis_manifest.normalize_text
    (strip, lowercase, collapse whitespace), then split at the
    deterministic character midpoint mid = len(normalized) // 2 (floor
    division): fragment_a = normalized[:mid], fragment_b =
    normalized[mid:]. Odd-length rule: fragment_a always gets
    floor(L/2) characters (the shorter or equal half), fragment_b
    always gets ceil(L/2) characters (the longer or equal half, absorbing
    the middle character when L is odd).

    Reconstruction (fragment_a + fragment_b) reproduces the NORMALIZED
    instruction, not necessarily the raw original byte-for-byte (they
    differ only in whitespace/case) -- this is the frozen, deliberate
    choice per this round's instruction, not an oversight.

    Returns (fragment_a, fragment_b, normalized_sha256, fragment_a_sha256, fragment_b_sha256).
    normalized_sha256 is identical to axis_manifest.normalized_text_hash(instruction_text).
    """
    normalized = normalize_text(instruction_text)
    mid = len(normalized) // 2
    fragment_a = normalized[:mid]
    fragment_b = normalized[mid:]
    assert fragment_a + fragment_b == normalized
    return (
        fragment_a,
        fragment_b,
        normalized_text_hash(instruction_text),
        sha256_hex(fragment_a),
        sha256_hex(fragment_b),
    )


if __name__ == "__main__":
    for sample in ["Example instruction.", "Odd length example!", "café résumé"]:
        b64, in_h, out_h = encode_base64_utf8(sample)
        print(f"encode_base64_utf8({sample!r}) -> {b64!r} input_sha256={in_h[:12]}.. output_sha256={out_h[:12]}..")
    for sample in ["Even length instr.", "An odd length one.", "  Extra   whitespace  Mixed CASE  "]:
        a, b, norm_h, a_h, b_h = split_payload_midpoint(sample)
        print(f"split_payload_midpoint({sample!r}) -> a={a!r} b={b!r} "
              f"len(a)={len(a)} len(b)={len(b)} normalized_sha256={norm_h[:12]}..")
