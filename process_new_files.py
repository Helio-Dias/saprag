#!/usr/bin/env python3
"""Process all .doc files in the arquivos_novos folder and write chunks to Firestore."""

import datetime
import email
import os
import quopri
import re
import urllib.parse
from html import unescape
from html.parser import HTMLParser

import fitz  # PyMuPDF
from google.cloud import firestore

GCP_PROJECT = "ems-data-platform-dev"
FIRESTORE_DATABASE = "helio-test-db"
FOLDER = os.path.join(os.path.dirname(__file__), "arquivos_novos")
CHUNK_SIZE = 528
OVERLAP = 128


def extract_pdf_text(path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return _normalize("\n".join(parts))

# tags whose entire content should be discarded (CSS, scripts, metadata)
_SKIP_TAGS = {"style", "script", "head", "xml"}

# known SQL placeholders that contain < but are not HTML
# detects real HTML tags via the whitelist
_HTML_TAGS = {
    "div","p","span","a","b","i","strong","em","ul","ol","li",
    "table","tr","td","th","thead","tbody","br","hr","h1","h2",
    "h3","h4","h5","h6","img","code","pre","blockquote","section",
    "article","header","footer","nav","form","input","button",
}


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML, ignoring <head>, <style>, <script>."""

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1
        # self-closing block tags that have no end tag
        if tag.lower() in {"br", "hr"} and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        # paragraph / block → newline
        if tag.lower() in {"p","div","li","br","tr","h1","h2","h3","h4","h5","h6"}:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name):
        if self._skip_depth == 0:
            self._parts.append(unescape(f"&{name};"))

    def handle_charref(self, name):
        if self._skip_depth == 0:
            self._parts.append(unescape(f"&#{name};"))

    def get_text(self) -> str:
        return "".join(self._parts)


def extract_mime_text(raw: str) -> str:
    """Extract plain text from a multipart MIME file exported from Confluence."""
    msg = email.message_from_string(raw)
    html_part = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=False)
                enc = part.get("Content-Transfer-Encoding", "").lower()
                if enc == "quoted-printable":
                    payload = quopri.decodestring(
                        payload.encode("latin-1") if isinstance(payload, str) else payload
                    ).decode("utf-8", errors="replace")
                html_part = payload
                break
    else:
        ct = msg.get_content_type()
        if ct == "text/html":
            payload = msg.get_payload(decode=False)
            enc = msg.get("Content-Transfer-Encoding", "").lower()
            if enc == "quoted-printable":
                payload = quopri.decodestring(
                    payload.encode("latin-1") if isinstance(payload, str) else payload
                ).decode("utf-8", errors="replace")
            html_part = payload

    if html_part is None:
        # not MIME — treat as plain text
        return _clean_plain_text(raw)

    parser = _TextExtractor()
    parser.feed(html_part)
    text = parser.get_text()
    return _normalize(text)


def _clean_plain_text(text: str) -> str:
    """Fallback for non-MIME files — strip HTML if present."""
    # decode residual quoted-printable
    text = re.sub(r"=\r?\n", "", text)
    text = re.sub(r"=([0-9A-Fa-f]{2})",
                  lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), text)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:[^>]*)?>", " ", text)
    text = unescape(text)
    return _normalize(text)


def _normalize(text: str) -> str:
    # remove any <br> or <br/> that survived as literal text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_no_html(chunks: list[str]) -> list[str]:
    """Return chunks with real HTML tags (excludes SQL placeholders and index names)."""
    def has_html(c: str) -> bool:
        for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)(?:[^>]*)?>", c):
            name = m.group(2).lower()
            if name not in _HTML_TAGS:
                continue
            # check context: if preceded by _ it is an index placeholder (idx_<table>)
            start = m.start()
            if start > 0 and c[start - 1] == "_":
                continue
            return True
        return False
    return [c for c in chunks if has_html(c)]


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []

    chunks = []
    pos = 0
    length = len(text)

    while pos < length:
        end = pos + chunk_size

        if end >= length:
            segment = text[pos:].strip()
            if segment:
                chunks.append(segment)
            break

        while end < length and not text[end].isspace():
            end += 1

        segment = text[pos:end].strip()
        if segment:
            chunks.append(segment)

        next_pos = end - overlap
        if next_pos < length and next_pos > 0 and not text[next_pos].isspace():
            while next_pos < length and not text[next_pos].isspace():
                next_pos += 1
        while next_pos < length and text[next_pos].isspace():
            next_pos += 1
        if next_pos <= pos:
            next_pos = end
            while next_pos < length and text[next_pos].isspace():
                next_pos += 1

        pos = next_pos

    return chunks


def get_next_id(db: firestore.Client) -> int:
    docs = list(db.collection("documentos").stream())
    if not docs:
        return 1
    return max(int(d.id) for d in docs) + 1


def write_to_firestore(db: firestore.Client, doc_id: int, description: str, path: str, chunks: list[str]):
    doc_ref = db.collection("documentos").document(str(doc_id))
    # create() fails if the id already exists, preventing a stale next_id from
    # silently overwriting an existing document.
    doc_ref.create({
        "id": doc_id,
        "descricao": description,
        "caminho_arquivo": path,
        "total_chunks": len(chunks),
        "data_onboard": datetime.datetime.now(datetime.timezone.utc),
    })

    for i in range(0, len(chunks), 100):
        batch = db.batch()
        for seq, chunk in enumerate(chunks[i:i + 100], i + 1):
            chunk_ref = doc_ref.collection("chunks").document(str(seq))
            batch.set(chunk_ref, {
                "id_documento": doc_id,
                "sequencia_chunk": seq,
                "texto": chunk,
                "embedding": None,
            })
        batch.commit()


def description_from_name(file_name: str) -> str:
    without_ext = os.path.splitext(file_name)[0]
    return urllib.parse.unquote(without_ext).replace("+", " ")


def process():
    files = sorted(
        f for f in os.listdir(FOLDER)
        if f.lower().endswith((".doc", ".docx", ".txt", ".md", ".pdf"))
    )

    if not files:
        print("No .doc/.docx/.txt/.md/.pdf files found in arquivos_novos/")
        return

    print(f"Found {len(files)} file(s) to process.\n")
    db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)

    # paths already written to Firestore — avoids duplicates
    already_written = {
        d.to_dict().get("caminho_arquivo")
        for d in db.collection("documentos").stream()
    }

    next_id = get_next_id(db)
    total_chunks = 0
    errors = []

    for i, name in enumerate(files, 1):
        path = os.path.join(FOLDER, name)
        description = description_from_name(name)
        print(f"[{i}/{len(files)}] {description}")

        if path in already_written:
            print(f"  Already exists in Firestore, skipping.")
            continue
        try:
            raw = read_file(path)
            if name.lower().endswith(".pdf"):
                text = extract_pdf_text(path)
            elif name.lower().endswith(".md"):
                text = _normalize(raw)
            else:
                text = extract_mime_text(raw)
        except Exception as e:
            print(f"  ERROR while processing: {e}\n")
            errors.append(name)
            continue

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        if not chunks:
            print(f"  WARNING: empty file after cleanup, skipping.")
            errors.append(name)
            continue

        dirty = validate_no_html(chunks)
        if dirty:
            print(f"  ERROR: {len(dirty)} chunk(s) still contain HTML. Skipping file.")
            print(f"  Example: {dirty[0][:150]}")
            errors.append(name)
            continue

        write_to_firestore(db, next_id, description, path, chunks)
        print(f"  Document ID {next_id} — {len(chunks)} chunk(s) written.")
        total_chunks += len(chunks)
        next_id += 1

    print(f"\nDone. {len(files) - len(errors)}/{len(files)} document(s), {total_chunks} chunk(s) written to Firestore.")
    if errors:
        print(f"Files with errors ({len(errors)}): {', '.join(e[:40] for e in errors)}")


if __name__ == "__main__":
    process()
