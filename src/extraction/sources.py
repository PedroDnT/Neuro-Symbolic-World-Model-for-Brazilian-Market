"""Find, download and read the full filing documents of the CVM document server.

2 document types are read:

    FRE 2021  The download is a zip that holds a second zip (".fre"). Its
              AnexoDocumento.xml holds each free-text item of the form as a base64 PDF.
              Group 828 is section 7 of the form (activities of the company), which holds
              the supplier and customer items.
    DFP 2024  The download is a zip that holds 1 PDF with the statements and the notes.

A locator keeps only the text that matches the task keywords, so the model reads a few
pages and not the full document.
"""
from __future__ import annotations

import base64
import functools
import io
import json
import re
import threading
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

from src.data_ingestion.download import CVM_BASE, RAW_DIR, download
from src.protocol import data_fetch_error

MODULE = "src/extraction/sources.py"
ACTIVITY_GROUP = "828"
MAX_CHARS = 60_000
LONG_SECTION_CHARS = 20_000
WINDOW_RADIUS = 1_500
MAX_DEBT_PAGES = 8
MIN_INDEXER_HITS = 2

SUPPLIER_PATTERN = re.compile(
    r"fornecedor|mat[ée]rias?[- ]primas?|insumos|clientes? respons|10% da receita|principais clientes", re.I
)
DEBT_PATTERN = re.compile(r"empr[ée]stimos|financiamentos|deb[êe]ntures", re.I)
INDEXER_PATTERN = re.compile(
    r"\bCDI\b|\bIPCA\b|\bSELIC\b|\bTJLP\b|\bTLP\b|\bSOFR\b|\bLIBOR\b|pr[ée][- ]?fixad|indexador|% ?a\.a\.", re.I
)

# PDFium is not safe when 2 threads use it at the same time.
_PDF_LOCK = threading.Lock()


@dataclass(frozen=True)
class Section:
    location: str
    text: str


@functools.lru_cache(maxsize=None)
def _index(doc: str, year: int) -> pd.DataFrame:
    name = f"{doc}_cia_aberta_{year}.zip"
    path = download(f"{CVM_BASE}/DOC/{doc.upper()}/DADOS/{name}", RAW_DIR / "cvm" / name, MODULE)
    with zipfile.ZipFile(path) as archive, archive.open(f"{doc}_cia_aberta_{year}.csv") as handle:
        frame = pd.read_csv(handle, sep=";", encoding="latin1", dtype=str)
    return frame.assign(cd_cvm=frame["CD_CVM"].astype(int), version=frame["VERSAO"].astype(int))


def document_link(doc: str, year: int, cd_cvm: int) -> dict | None:
    """The last filed version of a document of 1 company. None when the index has none."""
    rows = _index(doc, year)
    rows = rows[rows["cd_cvm"] == cd_cvm].sort_values(["DT_REFER", "version"])
    if rows.empty:
        return None
    last = rows.iloc[-1]
    return {
        "doc": doc,
        "year": year,
        "cd_cvm": cd_cvm,
        "id_doc": str(last["ID_DOC"]),
        "version": int(last["version"]),
        "reference_date": str(last["DT_REFER"]),
        "link": str(last["LINK_DOC"]),
    }


def fetch(document: dict) -> Path:
    """Download the document zip into the cache. Raise ERR_DATA_FETCH when it is not a zip."""
    path = RAW_DIR / "rad" / f"{document['doc']}_{document['id_doc']}.zip"
    download(document["link"], path, MODULE)
    if not zipfile.is_zipfile(path):
        path.unlink()
        raise data_fetch_error(MODULE, zipfile.BadZipFile(document["link"]))
    return path


def _pdf_pages(data: bytes) -> list[str]:
    """Text of each page. A PDF that PDFium cannot open gives no page."""
    with _PDF_LOCK:
        try:
            document = pdfium.PdfDocument(data)
        except pdfium.PdfiumError:
            return []
        try:
            pages = []
            for index in range(len(document)):
                page = document[index]
                text_page = page.get_textpage()
                pages.append(text_page.get_text_range())
                text_page.close()
                page.close()
            return pages
        finally:
            document.close()


def fre_sections(path: Path, group: str = ACTIVITY_GROUP) -> list[Section]:
    """The free-text items of 1 group of the 2021 reference form."""
    with zipfile.ZipFile(path) as outer:
        inner_name = next((name for name in outer.namelist() if name.lower().endswith(".fre")), None)
        if inner_name is None:
            return []
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            if "AnexoDocumento.xml" not in inner.namelist():
                return []
            root = ET.fromstring(inner.read("AnexoDocumento.xml"))
    sections = []
    for item in root:
        fields = {child.tag.split("}")[-1]: child.text or "" for child in item.iter()}
        if fields.get("NumeroGrupoRelacionado") != group:
            continue
        text = "\n".join(_pdf_pages(base64.b64decode(fields.get("ImagemObjetoArquivoPdf", "")))).strip()
        if text:
            sections.append(Section(f"FRE 2021, section 7, field {fields.get('NumeroQuadroRelacionado')}", text))
    return sections


def dfp_pages(path: Path) -> list[Section]:
    """The pages of the statement PDF. The page text is cached next to the zip."""
    cache = path.with_suffix(".pages.json")
    if cache.exists():
        return [Section(**item) for item in json.loads(cache.read_text())]
    with zipfile.ZipFile(path) as outer:
        pdfs = [name for name in outer.namelist() if name.lower().endswith(".pdf")]
        if not pdfs:
            return []
        data = outer.read(max(pdfs, key=lambda name: outer.getinfo(name).file_size))
    pages = [
        Section(f"DFP 2024, page {number}", text)
        for number, text in enumerate(_pdf_pages(data), start=1)
        if text.strip()
    ]
    cache.write_text(json.dumps([asdict(page) for page in pages]))
    return pages


def _within_budget(ranked: list[Section], order: list[Section]) -> list[Section]:
    """Take sections in rank order until the character budget is full. Return them in document order."""
    kept, used = set(), 0
    for section in ranked:
        if used + len(section.text) > MAX_CHARS and kept:
            break
        kept.add(section.location)
        used += len(section.text)
    return [section for section in order if section.location in kept]


def _keyword_windows(text: str, pattern: re.Pattern) -> str:
    """For a long text, keep only the parts around each keyword hit."""
    if len(text) <= LONG_SECTION_CHARS:
        return text
    spans: list[list[int]] = []
    for match in pattern.finditer(text):
        start, end = max(0, match.start() - WINDOW_RADIUS), min(len(text), match.end() + WINDOW_RADIUS)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return "\n[...]\n".join(text[start:end] for start, end in spans)


def locate_supplier_sections(sections: list[Section]) -> list[Section]:
    hits = {section.location: len(SUPPLIER_PATTERN.findall(section.text)) for section in sections}
    trimmed = [
        Section(section.location, _keyword_windows(section.text, SUPPLIER_PATTERN))
        for section in sections
        if hits[section.location]
    ]
    ranked = sorted(trimmed, key=lambda section: -hits[section.location])
    return _within_budget(ranked, trimmed)


def locate_debt_pages(pages: list[Section]) -> list[Section]:
    def score(page: Section) -> int:
        return len(INDEXER_PATTERN.findall(page.text)) if DEBT_PATTERN.search(page.text) else 0

    scores = {page.location: score(page) for page in pages}
    ranked = sorted((p for p in pages if scores[p.location] >= MIN_INDEXER_HITS), key=lambda p: -scores[p.location])
    return _within_budget(ranked[:MAX_DEBT_PAGES], pages)
