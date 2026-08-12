import io
import json
import re
import base64
import posixpath
import zipfile
import pandas as pd
from lxml import etree
from dotenv import load_dotenv

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS

from pypdf import PdfReader
from docx import Document as DocxDocument
from docx.enum.text import WD_COLOR_INDEX
from pathlib import Path
from docx.oxml.ns import qn

BASE_DIR = Path(__file__).resolve().parent
AOR_TEMPLATE_PATH = BASE_DIR / "templates" / "Sample AOR Template.docx"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "version": "test-commit-1"}


# =========================================================
# EXTRACTION FUNCTIONS
# =========================================================

def extract_txt(contents: bytes, filename: str):
    text = contents.decode("utf-8", errors="ignore")
    return [
        LCDocument(
            page_content=text,
            metadata={
                "filename": filename,
                "source_type": "txt",
                "content_type": "plain_text"
            }
        )
    ]


def extract_pdf(contents: bytes, filename: str):
    docs = []
    try:
        pdf_stream = io.BytesIO(contents)
        pdf_reader = PdfReader(pdf_stream)
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                docs.append(
                    LCDocument(
                        page_content=page_text,
                        metadata={
                            "filename": filename,
                            "source_type": "pdf",
                            "page": page_number,
                            "content_type": "page_text"
                        }
                    )
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    return docs

RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _strip_dangling_relationships(contents: bytes) -> bytes:
    """Remove .rels entries pointing at parts missing from the archive.

    Some tools re-save .docx files without cleaning up relationship
    references (e.g. to word/people.xml for @mentions), which makes
    python-docx raise a KeyError when it tries to walk every part.
    """
    zin = zipfile.ZipFile(io.BytesIO(contents))
    names = set(zin.namelist())
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename.endswith(".rels"):
                base_dir = posixpath.dirname(posixpath.dirname(item.filename))
                tree = etree.fromstring(data)

                for rel in tree.findall(f"{{{RELS_NS}}}Relationship"):
                    if rel.get("TargetMode") == "External":
                        continue
                    target = posixpath.normpath(
                        posixpath.join(base_dir, rel.get("Target"))
                    )
                    if target not in names:
                        tree.remove(rel)

                data = etree.tostring(
                    tree, xml_declaration=True, encoding="UTF-8", standalone=True
                )

            zout.writestr(item, data)

    return buffer.getvalue()

def build_result_docx(result_text: str, inputs: dict = None, metrics: dict = None) -> bytes:
    """
    Populate the existing AOR template sections using the drafted AOR text.
    The template's layout, tables, headers, footers and annexes are retained.
    """

    doc = DocxDocument(str(AOR_TEMPLATE_PATH))

    SECTION_ALIASES = {
        "purpose": "purpose",
        "background": "background",
        "need for deployment": "need",
        "need for deployment of the sample system": "need",
        "scope of work": "scope",
        "estimated costs": "costs",
        "proposed budget": "costs",
        "net economic value (nev) analysis and manpower capitalisation": "nev",
        "net economic value analysis and manpower capitalisation": "nev",
        "funding": "funding",
        "availability of funds": "funding",
        "approving authority": "authority",
        "approval": "approval",
    }

    def normalise_heading(text: str) -> str:
        text = re.sub(r"^\d+[\.\s]+", "", text.strip())
        text = text.replace("**", "")
        text = text.rstrip(":")
        return re.sub(r"\s+", " ", text).strip().lower()

    def populate_budget_table(inputs: dict, metrics: dict):
        capex_items = inputs.get("capex_items", [])
        opex_items = inputs.get("opex_items", [])
        grand_total = inputs.get("grand_total", "")
        total_capex = metrics.get("total_capex", "")
        total_opex = metrics.get("total_opex", "")

        for table in doc.tables:
            # Check if this looks like the budget table by scanning headers
            header_text = " ".join(
                cell.text.strip().lower()
                for cell in table.rows[0].cells
            )
            if "description" not in header_text and "cost" not in header_text:
                continue

            # Find CAPEX and OPEX row ranges
            capex_start = None
            opex_start = None
            total_capex_row = None
            total_opex_row = None
            grand_total_row = None

            for i, row in enumerate(table.rows):
                first_cell = row.cells[0].text.strip().lower()
                if "capex" in first_cell and capex_start is None:
                    capex_start = i + 1
                elif "total capex" in first_cell:
                    total_capex_row = i
                elif "opex" in first_cell and opex_start is None:
                    opex_start = i + 1
                elif "total opex" in first_cell:
                    total_opex_row = i
                elif "grand total" in first_cell:
                    grand_total_row = i

            # Fill CAPEX rows
            if capex_start is not None:
                for j, item in enumerate(capex_items):
                    row_index = capex_start + j
                    if total_capex_row and row_index >= total_capex_row:
                        break
                    row = table.rows[row_index]
                    cells = row.cells
                    if len(cells) >= 3:
                        cells[0].text = str(j + 1)
                        cells[1].text = item.get("description", "")
                        cells[2].text = str(item.get("amount", ""))

            # Fill Total CAPEX
            if total_capex_row is not None:
                cells = table.rows[total_capex_row].cells
                if len(cells) >= 3:
                    cells[2].text = str(total_capex) if total_capex else ""

            # Fill OPEX rows
            if opex_start is not None:
                for j, item in enumerate(opex_items):
                    row_index = opex_start + j
                    if total_opex_row and row_index >= total_opex_row:
                        break
                    row = table.rows[row_index]
                    cells = row.cells
                    if len(cells) >= 3:
                        cells[0].text = str(len(capex_items) + j + 1)
                        cells[1].text = item.get("description", "")
                        cells[2].text = str(item.get("amount", ""))

            # Fill Total OPEX
            if total_opex_row is not None:
                cells = table.rows[total_opex_row].cells
                if len(cells) >= 3:
                    cells[2].text = str(total_opex) if total_opex else ""

            # Fill Grand Total
            if grand_total_row is not None:
                cells = table.rows[grand_total_row].cells
                if len(cells) >= 3:
                    cells[2].text = str(grand_total) if grand_total else ""

    def parse_aor_sections(text: str):
        sections = {}
        title_lines = []
        current_section = None
        current_lines = []

        def save_current_section():
            if current_section and current_lines:
                sections[current_section] = "\n".join(
                    line for line in current_lines if line.strip()
                ).strip()

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalised = normalise_heading(line)
            detected_section = SECTION_ALIASES.get(normalised)

            if detected_section:
                save_current_section()
                current_section = detected_section
                current_lines = []
                continue

            if current_section is None:
                title_lines.append(line)
            else:
                current_lines.append(line)

        save_current_section()

        title = title_lines[0] if title_lines else ""
        title = title.replace("**", "").strip()

        return title, sections

    def replace_paragraph_text(paragraph, new_text: str):
        """
        Replace paragraph contents while retaining the paragraph formatting.
        """
        paragraph_element = paragraph._p

        for child in list(paragraph_element):
            if child.tag == qn("w:pPr"):
                continue
            paragraph_element.remove(child)

        lines = new_text.splitlines()

        for index, line in enumerate(lines):
            if index > 0:
                paragraph.add_run().add_break()

            cleaned_line = re.sub(r"^[-*]\s+", "", line.strip())
            paragraph.add_run(cleaned_line)

    def find_heading_index(heading_names):
        names = {
            normalise_heading(name)
            for name in heading_names
        }

        for index, paragraph in enumerate(doc.paragraphs):
            paragraph_heading = normalise_heading(paragraph.text)

            if paragraph_heading in names:
                return index

        return None

    def replace_section_body(heading_names, new_text: str):
        if not new_text:
            return

        heading_index = find_heading_index(heading_names)

        if heading_index is None:
            return

        paragraphs = doc.paragraphs
        first_content_paragraph = None  # <-- move this OUTSIDE the loop

        for index in range(heading_index + 1, len(paragraphs)):
            paragraph = paragraphs[index]
            paragraph_text = paragraph.text.strip()

            if not paragraph_text:
                continue

            normalised = normalise_heading(paragraph_text)

            if normalised in SECTION_ALIASES:
                return

            if first_content_paragraph is None:
                first_content_paragraph = paragraph
                replace_paragraph_text(paragraph, new_text)
            else:
                paragraph.text = ""  # clear subsequent paragraphs in the section"

    title, sections = parse_aor_sections(result_text)

    # Replace the template title.
    if title:
        for paragraph in doc.paragraphs:
            paragraph_text = paragraph.text.strip()

            if "<TITLE>" in paragraph_text.upper():
                replace_paragraph_text(
                    paragraph,
                    title.upper()
                )
                break

    # Populate each corresponding template section.
    replace_section_body(
        ["Purpose"],
        sections.get("purpose", "")
    )

    replace_section_body(
        ["Background"],
        sections.get("background", "")
    )

    replace_section_body(
        [
            "Need for <xx>",
            "Need For Deployment",
            "Need for Deployment of the Sample System"
        ],
        sections.get("need", "")
    )

    replace_section_body(
        ["Scope of Work"],
        sections.get("scope", "")
    )

    replace_section_body(
        ["Proposed Budget", "Estimated Costs"],
        sections.get("costs", "")
    )

    replace_section_body(
        [
            "Net Economic Value (NEV) Analysis and Manpower Capitalisation",
            "Net Economic Value Analysis and Manpower Capitalisation"
        ],
        sections.get("nev", "")
    )

    replace_section_body(
        ["Availability of Funds", "Funding"],
        sections.get("funding", "")
    )

    replace_section_body(
        ["Approving Authority"],
        sections.get("authority", "")
    )

    replace_section_body(
        ["Approval"],
        sections.get("approval", "")
    )

    # Populate budget table if structured data is available
    if inputs and metrics:
        populate_budget_table(inputs, metrics)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def _load_docx(contents: bytes) -> DocxDocument:
    try:
        return DocxDocument(io.BytesIO(contents))
    except KeyError:
        return DocxDocument(io.BytesIO(_strip_dangling_relationships(contents)))


def extract_docx(contents: bytes, filename: str):
    docs = []
    try:
        doc = _load_docx(contents)

        paragraphs = [
            para.text.strip()
            for para in doc.paragraphs
            if para.text.strip()
        ]
        if paragraphs:
            docs.append(
                LCDocument(
                    page_content="\n".join(paragraphs),
                    metadata={
                        "filename": filename,
                        "source_type": "docx",
                        "content_type": "paragraphs"
                    }
                )
            )
        for table_index, table in enumerate(doc.tables, start=1):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                docs.append(
                    LCDocument(
                        page_content="\n".join(rows),
                        metadata={
                            "filename": filename,
                            "source_type": "docx",
                            "content_type": "table",
                            "table_index": table_index
                        }
                    )
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Word Document: {str(e)}")
    return docs


def extract_candidate_headings(contents: bytes) -> list:
    """Pull short, standalone paragraphs and table rows that likely function
    as section headings. Many AOR documents lay out numbered section titles
    as a two-column table row (e.g. "10 | Scope of Work") rather than a
    plain paragraph, and bold a heading rather than apply Word's built-in
    Heading style — so paragraph.style alone isn't reliable. Length is used
    as a rough heuristic instead, and the LLM is left to judge which
    candidates are actually headings.
    """
    doc = _load_docx(contents)
    candidates = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text and len(text) <= 80:
            candidates.append(text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text and len(row_text) <= 80:
                candidates.append(row_text)
    return candidates


def extract_excel(contents: bytes, filename: str):
    docs = []
    try:
        excel = pd.read_excel(io.BytesIO(contents), sheet_name=None)
        for sheet_name, df in excel.items():
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            docs.append(
                LCDocument(
                    page_content=df.to_csv(index=False),
                    metadata={
                        "filename": filename,
                        "source_type": "excel",
                        "sheet": sheet_name,
                        "content_type": "sheet_table",
                        "rows": len(df),
                        "columns": list(df.columns)
                    }
                )
            )
            for idx, row in df.iterrows():
                row_dict = row.dropna().to_dict()
                if row_dict:
                    docs.append(
                        LCDocument(
                            page_content=json.dumps(row_dict, ensure_ascii=False),
                            metadata={
                                "filename": filename,
                                "source_type": "excel",
                                "sheet": sheet_name,
                                "content_type": "row",
                                "row_index": idx + 1
                            }
                        )
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Excel Document: {str(e)}")
    return docs


def extract_documents(contents: bytes, filename: str):
    filename_lower = filename.lower()
    if filename_lower.endswith(".txt"):
        return extract_txt(contents, filename)
    if filename_lower.endswith(".pdf"):
        return extract_pdf(contents, filename)
    if filename_lower.endswith(".docx"):
        return extract_docx(contents, filename)
    if filename_lower.endswith((".xlsx", ".xls")):
        return extract_excel(contents, filename)
    raise HTTPException(
        status_code=400,
        detail="Unsupported file format. Please upload .txt, .pdf, .docx, .xlsx, or .xls files."
    )


# =========================================================
# SPLITTING
# =========================================================

def split_documents(raw_docs, embeddings):
    splitter = SemanticChunker(embeddings)
    result = []
    for doc in raw_docs:
        # Tables (and Excel rows) must stay intact — splitting them can
        # separate a section header (e.g. "CAPEX"/"OPEX") from the rows/
        # totals underneath it, causing the LLM to mislabel which figure
        # belongs to which section.
        if doc.metadata.get("content_type") in ("row", "table"):
            result.append(doc)
            continue
        chunks = splitter.create_documents(
            [doc.page_content],
            metadatas=[doc.metadata]
        )
        for i, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_index"] = i
            result.append(chunk)
    return result


# =========================================================
# JSON AND NUMBER HELPERS
# =========================================================

def parse_json_from_llm(text: str):
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON object found in LLM response.")


def to_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in ["not found", "not found in document", "na", "n/a", "none", "null", ""]:
            return None
        cleaned = cleaned.replace("S$", "").replace("$", "").replace(",", "").replace("%", "")
        match = re.search(r"-?\d+(\.\d+)?", cleaned)
        if match:
            number = float(match.group(0))
            return int(number) if number.is_integer() else number
    return None


def clean_extracted_inputs(inputs: dict):
    numeric_fields = [
        "capex",
        "opex",
        "project_duration_years",
        "annual_productivity_time_savings_hours",
        "annual_manpower_impact_fte",
        "annual_benefit",
        "num_staff",
        "savings_duration_months",
        "man_hour_rate",
        "grand_total"
    ]
    cleaned = dict(inputs)
    for field in numeric_fields:
        cleaned[field] = to_number(cleaned.get(field))
    return cleaned


# =========================================================
# ICT AOR CALCULATIONS
# =========================================================

def compute_aor_metrics(inputs: dict):
    metrics = {}

    capex = inputs.get("capex")
    opex = inputs.get("opex")
    project_duration_years = inputs.get("project_duration_years")
    annual_productivity_time_savings_hours = inputs.get("annual_productivity_time_savings_hours")
    annual_manpower_impact_fte = inputs.get("annual_manpower_impact_fte")
    annual_benefit = inputs.get("annual_benefit")
    savings_duration_months = inputs.get("savings_duration_months")
    man_hour_rate = inputs.get("man_hour_rate")

    # --- Sanity log ---
    metrics["debug_project_duration_years"] = project_duration_years  # 👈 here
    metrics["debug_capex"] = capex
    metrics["debug_opex"] = opex

    hours_per_fte = 1819
    r = 0.06  # 4% nominal discount + 2% inflation

    # --- PV factor (annuity, computed from project duration) ---
    if project_duration_years is not None and project_duration_years > 0:
        pv_factor = (1 - (1 + r) ** -project_duration_years) / r
        metrics["pv_factor"] = round(pv_factor, 4)
    else:
        metrics["pv_factor"] = None

    # --- Annual OPEX (OPEX spread evenly over project duration) ---
    if opex is not None and project_duration_years is not None and project_duration_years > 0:
        metrics["annual_opex"] = round(opex / project_duration_years, 2)
    else:
        metrics["annual_opex"] = None

    # --- Annualised productivity hours ---
    if annual_productivity_time_savings_hours is not None:
        duration_months = savings_duration_months if savings_duration_months else 12
        metrics["annualised_productivity_hours"] = round(
            annual_productivity_time_savings_hours * (12 / duration_months), 2
        )
    else:
        metrics["annualised_productivity_hours"] = None

    # --- Annual manpower impact (FTE) ---
    if annual_manpower_impact_fte is not None:
        metrics["annual_manpower_impact_fte"] = annual_manpower_impact_fte
    elif metrics["annualised_productivity_hours"] is not None:
        metrics["annual_manpower_impact_fte"] = round(
            metrics["annualised_productivity_hours"] / hours_per_fte, 4
        )
    else:
        metrics["annual_manpower_impact_fte"] = None

    # --- Annual benefit ---
    if annual_benefit is not None:
        metrics["annual_benefit"] = annual_benefit
    elif metrics["annual_manpower_impact_fte"] is not None and man_hour_rate is not None:
        metrics["annual_benefit"] = round(
            metrics["annual_manpower_impact_fte"] * hours_per_fte * man_hour_rate, 2
        )
    else:
        metrics["annual_benefit"] = None

    # --- Total cost (simple sum, for reference only) ---
    if capex is not None and opex is not None:
        metrics["total_cost"] = round(capex + opex, 2)
    elif opex is not None:
        metrics["total_cost"] = round(opex, 2)
    else:
        metrics["total_cost"] = None

    # --- Total benefits PV (annual benefit discounted as annuity) ---
    if metrics["annual_benefit"] is not None and metrics["pv_factor"] is not None:
        metrics["total_benefits_pv"] = round(
            metrics["annual_benefit"] * metrics["pv_factor"], 2
        )
    else:
        metrics["total_benefits_pv"] = None

    # --- Total costs PV ---
    # CAPEX is already a present value (one-off upfront cost)
    # OPEX is treated as an annuity discounted over project duration
    if capex is not None and metrics["annual_opex"] is not None and metrics["pv_factor"] is not None:
        metrics["total_costs_pv"] = round(
            capex + (metrics["annual_opex"] * metrics["pv_factor"]), 2
        )
    else:
        metrics["total_costs_pv"] = None

    # --- Net Present Value ---
    if metrics["total_benefits_pv"] is not None and metrics["total_costs_pv"] is not None:
        metrics["net_present_value"] = round(
            metrics["total_benefits_pv"] - metrics["total_costs_pv"], 2
        )
    else:
        metrics["net_present_value"] = None

    # --- Benefit-Cost Ratio ---
    if (
        metrics["total_benefits_pv"] is not None
        and metrics["total_costs_pv"] is not None
        and metrics["total_costs_pv"] != 0
    ):
        metrics["benefit_cost_ratio"] = round(
            metrics["total_benefits_pv"] / metrics["total_costs_pv"], 4
        )
    else:
        metrics["benefit_cost_ratio"] = None

    return metrics

def identify_missing_fields(inputs: dict, metrics: dict):
    missing = []

    return missing

def determine_approving_authority(amount):
    if amount is None:
        return None

    if amount <= 6000:
        return (
            "Deputy Director / Senior Assistant Director",
            "Up to S$6,000"
        )

    elif amount <= 100000:
        return (
            "Division Head",
            "Up to S$100,000"
        )

    elif amount <= 250000:
        return (
            "Senior Director",
            "Up to S$250,000"
        )

    elif amount <= 500000:
        return (
            "Assistant/Deputy Director-General",
            "Up to S$500,000"
        )

    elif amount <= 1000000:
        return (
            "Director-General",
            "Up to S$1 million"
        )

    elif amount <= 5000000:
        return (
            "Management Committee",
            "Up to S$5 million"
        )

    elif amount <= 10000000:
        return (
            "Chairman",
            "Up to S$10 million"
        )

    else:
        return (
            "Authority",
            "Above S$10 million"
        )
# =========================================================
# DOCUMENT AMENDMENT
# =========================================================

    # These must come from the document — cannot be derived
    doc_required = ["capex", "opex", "project_duration_years"]
    for field in doc_required:
        if inputs.get(field) is None:
            missing.append(field)

    # These can be derived — only flag as missing if computation also failed
    derived_required = {
        "annual_manpower_impact_fte": metrics.get("annual_manpower_impact_fte"),
        "annual_benefit": metrics.get("annual_benefit"),
    }
    for field, computed_value in derived_required.items():
        if inputs.get(field) is None and computed_value is None:
            missing.append(field)

    return missing


# =========================================================
# DOCUMENT AMENDMENT
# =========================================================

FIELD_LABELS = {
    "capex": "Capital Expenditure (CAPEX)",
    "opex": "Operating Expenditure (OPEX)",
    "project_duration_years": "Project Duration (years)",
    "annual_manpower_impact_fte": "Annual Manpower Impact (FTE)",
    "annual_benefit": "Annual Benefit ($)",
}

FIELD_PLACEHOLDERS = {
    "capex": "[CAPEX — enter the total capital expenditure, e.g. S$500,000]",
    "opex": "[OPEX — enter the total operating expenditure over the project duration, e.g. S$300,000]",
    "project_duration_years": "[Project Duration — enter the number of years, e.g. 3]",
    "annual_manpower_impact_fte": "[Annual Manpower Impact (FTE) — enter the estimated FTE impact, e.g. 0.5]",
    "annual_benefit": "[Annual Benefit — enter the estimated annual dollar benefit, e.g. S$50,000]",
}


def build_missing_information_section(missing_fields: list) -> str:
    """Deterministically render the "Missing Information" section, rather
    than leaving its wording/formatting to the LLM each time.
    """
    if not missing_fields:
        return "5. Missing Information / Follow-up Required\n\nNo required information is missing."

    bullet_lines = "\n".join(
        f"- {FIELD_LABELS.get(field, field.replace('_', ' ').title())}"
        for field in missing_fields
    )
    return (
        "5. Missing Information / Follow-up Required\n\n"
        "You are missing the following required information. Please include this in your AOR:\n"
        f"{bullet_lines}"
    )


def build_structure_review_section(missing_sections: list, order_note: str) -> str:
    """Deterministically render the "Document Structure Review" section
    from the LLM's structured missing_sections/order_note output, rather
    than trusting free-form narrative formatting each time.
    """
    if not missing_sections:
        text = "All standard AOR sections appear to be present."
    else:
        bullet_lines = "\n".join(f"- {section}" for section in missing_sections)
        text = (
            "You are missing the following sections. Please include this in your AOR:\n"
            f"{bullet_lines}"
        )
    if order_note:
        text += f"\n\n{order_note}"
    return "6. Document Structure Review\n\n" + text


def build_amended_docx(contents: bytes, missing_fields: list, missing_sections: list = None) -> bytes:
    """Append a "Missing Information" section with fill-in placeholders
    to the user's own uploaded document, so they can complete it in Word.
    """
    doc = _load_docx(contents)

    doc.add_page_break()
    try:
        doc.add_heading("Missing Information — Please Complete", level=1)
    except KeyError:
        # Document has no "Heading 1" style defined — fall back to bold text.
        heading_paragraph = doc.add_paragraph()
        heading_paragraph.add_run("Missing Information — Please Complete").bold = True
    doc.add_paragraph(
        "The fields below were not found in your submission. "
        "Please replace each highlighted instruction with the correct value."
    )

    for field in missing_fields:
        label = FIELD_LABELS.get(field, field.replace("_", " ").title())
        placeholder = FIELD_PLACEHOLDERS.get(field, f"[{label} — enter value here]")
        paragraph = doc.add_paragraph()
        paragraph.add_run(f"The {label} is missing — please include: ").bold = True
        placeholder_run = paragraph.add_run(placeholder)
        placeholder_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    for section in (missing_sections or []):
        paragraph = doc.add_paragraph()
        paragraph.add_run("Missing section — ").bold = True
        section_run = paragraph.add_run(f'please add a "{section}" section to your AOR.')
        section_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# =========================================================
# PROMPTS
# =========================================================

STANDARD_AOR_SECTIONS = [
    "Purpose",
    "Background",
    "Need For Deployment",
    "Scope of Work",
    "Estimated Costs",
    "Net Economic Value (NEV) Analysis and Manpower Capitalisation",
    "Funding",
    "Approval",
    "Annex A: Detailed Scope of Works and Project Timeline",
    "Annex B: Detailed Cost Breakdown and Assessment",
    "Annex C: Net Economic Value (NEV)",
    "Annex D: Finance Manual Reference",
]


def build_structure_review_prompt(candidate_headings: list, standard_sections: list) -> str:
    headings_text = "\n".join(f"- {h}" for h in candidate_headings)
    standard_text = "\n".join(f"- {s}" for s in standard_sections)
    return f"""
You are reviewing whether an ICT AOR (Approval of Requirements) Word document
follows the standard CAAS AOR structure.

Standard AOR sections expected (in this rough order):
{standard_text}

Short/standalone lines found in the uploaded document (candidate section
headings, in document order — not every line here is actually a heading,
use judgement to tell real section headings apart from other short text):
{headings_text}

For each standard section, decide whether it is present: only count it as
present if one of the candidate lines above is itself a heading/title for
that section (matched by meaning, not exact wording — e.g. "10 Scope of
Work" counts as "Scope of Work"). Do NOT count a section as present just
because related content is mentioned in passing elsewhere without its own
heading — a heading that has been removed, leaving only orphaned prose
behind, must still be reported as missing.

Also check whether the sections that ARE present appear in a reasonable
order relative to the standard list, or if something is clearly out of
place.

Return ONLY a valid JSON object with this exact structure:
{{
  "missing_sections": ["<standard section name>", ...],
  "order_note": "<one sentence if something is out of order, otherwise an empty string>"
}}

Do NOT comment on financial figures or data completeness — that is handled
separately. Focus only on document structure. Do not wrap the JSON in
markdown, and do not include any text outside the JSON object.
"""


def build_extraction_prompt(context: str):
    return f"""
You are analysing an ICT Approval of Requirements (AOR) submission using the CAAS AOR template.

Your task is to extract RAW INPUTS only.
Do NOT perform calculations.
Do NOT infer missing values.
Do NOT invent figures, benefits, assumptions, or dates.
Do NOT substitute a figure from a different field or category just because it seems plausible
(e.g. do not use an OPEX figure for CAPEX, or a total budget figure for either). Each field must
come from text explicitly labelled for that exact field.
Do NOT annualise any hours figures.

Return ONLY a valid JSON object.
Do not wrap the JSON in markdown.
Do not include explanations outside the JSON.

Return exactly this JSON structure:

{{
  "project_title": null,
  "purpose": null,
  "problem_statement": null,
  "capex": null,
  "opex": null,
  "project_duration_years": null,
  "annual_productivity_time_savings_hours": null,
  "num_staff": null,
  "savings_duration_months": null,
  "man_hour_rate": null,
  "annual_manpower_impact_fte": null,
  "annual_benefit": null,
  "grand_total": null,
  "key_assumptions": [],
  "source_evidence": []
}}

Field guidance:
- "project_title": title or name of the ICT project.
- "purpose": what the ICT project seeks to achieve.
- "problem_statement": the gap, pain point, or problem being addressed.
- "capex": one-off capital expenditure (implementation, hardware, cybersecurity, contingency).
  Use a figure only if it is either (a) directly labelled CAPEX or "Capital Expenditure" in the
  same sentence/phrase, or (b) a Sub Total / Grand Total / "Say" figure inside a table section
  that is clearly grouped under a CAPEX / Capital Expenditure header, even if that specific row
  does not repeat the word "CAPEX" itself. Prefer the Grand Total (inclusive of contingency) over
  a Sub Total if both are present. Do NOT use a figure from the OPEX section, a combined/overall
  total covering both CAPEX and OPEX together, a man-hour rate, or any other unrelated figure,
  even if it seems plausible. If a CAPEX section header exists but every cost cell under it is
  blank/empty, that means capex is null — do NOT reach into the OPEX section (or anywhere else)
  for a substitute number just because CAPEX itself has none. If no such CAPEX figure exists,
  use null.
- "opex": total operating expenditure over the project duration (licences, maintenance, cloud support).
  Use a figure only if it is either (a) directly labelled OPEX or "Operating Expenditure" in the
  same sentence/phrase, or (b) a Sub Total / Grand Total / "Say" figure inside a table section
  that is clearly grouped under an OPEX / Operating Expenditure header (this section's line items
  are often described as "operations and maintenance", "recurring", "licences", or "subscription"
  without repeating the word "OPEX" on every row). Prefer the Grand Total (inclusive of
  contingency) over a Sub Total if both are present. Do NOT use a figure from the CAPEX section,
  a combined/overall total covering both CAPEX and OPEX together, a man-hour rate, or any other
  unrelated figure, even if it seems plausible. If an OPEX section header exists but every cost
  cell under it is blank/empty, that means opex is null — do NOT reach into the CAPEX section (or
  anywhere else) for a substitute number just because OPEX itself has none. If no such OPEX
  figure exists, use null.
- "grand_total": overall project cost including contingency,
  only if explicitly stated in the document.
  Prefer values labelled:
  - Grand Total
  - Total Budget
  - Total Cost (including contingency).

Worked example (a cost table where the CAPEX section header exists but every cost cell under
it is blank, and the OPEX section below it has real figures):

  S/N | Description                                          | Estimated Cost ($)
      | CAPEX                                                |
  1   | Implementation of sample system and associated works |
  2   | Provision of cybersecurity services                  |
      | Sub Total                                            |
      | Contingency (5% of Sub Total)                        |
      | Grand Total                                          |
      | Say                                                  |
      | OPEX                                                 |
  1   | Provision of recurring subscription services         | 522,000
  2   | Operations and maintenance support services           | 884,000
      | Sub Total                                            | 1,406,000
      | Contingency (5% of Sub Total)                        | 70,300
      | Grand Total                                          | 1,476,300
      | Say                                                  | 1.5m

The ONLY correct extraction from this table is "capex": null and "opex": 1476300.
It would be WRONG to set "capex" to 1476300, 1406000, or any other number from the OPEX
section — the CAPEX section has no figures of its own, so capex must be null, even though
that means leaving it blank rather than filling in a plausible-looking nearby number.
- "project_duration_years": project duration in years as an integer.   If the document states a date range (e.g. FY25 to FY28), count the  number of full operational years, not the number of financial year 
  labels. For example, FY25 to FY27 = 3 years, FY25 to FY28 = 4 years.  Prefer an explicitly stated duration (e.g. "3-year contract") over  a derived date range if both are present..
- "annual_productivity_time_savings_hours": the RAW total hours figure as stated in the document. Do NOT annualise. Do NOT adjust for number of staff or duration.
- "num_staff": number of staff whose time savings are being measured (e.g. 2 project officers).
- "savings_duration_months": duration in months over which the raw hours figure was measured (e.g. 9 months). Extract as a number.
- "man_hour_rate": the man-hour rate in dollars per hour as explicitly stated in the document (e.g. 98). Do NOT default to any standard rate if not found.
- "annual_manpower_impact_fte": annual manpower impact in FTE, only if explicitly stated as a final FTE figure in the document.
- "annual_benefit": annual benefit in dollars, only if explicitly stated as a final dollar benefit figure in the document.
- "key_assumptions": assumptions explicitly stated in the document.
- "source_evidence": short evidence snippets from the context.

Rules:
- Use null for missing single-value fields.
- Use [] for missing list fields.
- Preserve figures and wording as far as possible.
- Do NOT calculate total benefits, total costs, NPV, BCR, annualised hours, or FTE yourself.

Context:
{context}
"""


def build_explanation_prompt(context: str, inputs: dict, metrics: dict, missing_fields: list):
    return f"""
You are reviewing an ICT Approval of Requirements (AOR) submission using the CAAS AOR template.

Use the extracted inputs and computed metrics below.
Do NOT recompute any numbers.
Do NOT invent missing information.
If a metric is null, explain that it could not be computed because the required inputs were not found.

Extracted inputs:
{json.dumps(inputs, indent=2, ensure_ascii=False)}

Computed metrics:
{json.dumps(metrics, indent=2, ensure_ascii=False)}

Missing fields:
{json.dumps(missing_fields, indent=2, ensure_ascii=False)}

Context:
{context}

Return the assessment in this structure:

1. Purpose and Problem Statement

2. Cost Inputs
- CAPEX
- OPEX (total over project duration)
- Project duration

3. Benefits
- Annual productivity time savings (hours)
- Annual manpower impact (FTE)
- Annual benefit ($)

4. Value Assessment
- Total benefits (PV_benefit)
- Total costs (PV_cost)
- Net present value
- Benefit-cost ratio
- Feedback on completeness of user's submission

Stop after section 4. Do NOT write a "Missing Information" section yourself —
it will be added separately from the authoritative missing fields list above.
"""

def build_aor_drafting_prompt(
    context: str,
    inputs: dict,
    metrics: dict,
    approval_info
):
    return f"""
You are drafting a complete CAAS ICT AOR paper.

Use the extracted inputs below.

Extracted Inputs:
{json.dumps(inputs, indent=2)}

Computed Metrics:
{json.dumps(metrics, indent=2)}

Approval Information:
{approval_info}

Source Context:
{context}

Draft the AOR using this structure:

TITLE

Purpose

Background

Need For Deployment

Scope of Work

Estimated Costs

Net Economic Value (NEV) Analysis and Manpower Capitalisation

Funding

Approving Authority

Approval

Rules:
- Write in formal CAAS paper style.
- Use the extracted figures.
- Use approval_info for the Approving Authority section.
- Do not create Annexes.
- Do not invent figures.
- Return ONLY the completed AOR text.
"""
def build_aor_drafting_prompt(
    context: str,
    inputs: dict,
    metrics: dict,
    approval_info
):
    return f"""
    
You are drafting a complete ICT AOR paper.

Use the extracted inputs below.

Extracted Inputs:
{json.dumps(inputs, indent=2, ensure_ascii=False)}

Computed Metrics:
{json.dumps(metrics, indent=2, ensure_ascii=False)}

Approval Information:
{approval_info}

Context:
{context}

Draft the AOR using this structure:

TITLE

Purpose

Background

Need For Deployment

Scope of Work

Estimated Costs

Net Economic Value (NEV) Analysis and Manpower Capitalisation

Funding

Approving Authority

Approval

Rules:
- Use formal CAAS paper writing style.
- Use figures from the extracted inputs.
- Use approval_info for the Approving Authority section.
- Do not create Annexes.
- Do not invent figures.
- Return only the AOR text.
- In the Net Economic Value section, state the following computed values exactly:
  - Present Value of Benefits: use total_benefits_pv
  - Present Value of Costs: use total_costs_pv
  - Net Present Value: use net_present_value
  - Benefit-Cost Ratio: use benefit_cost_ratio
  - Annual Manpower Impact: use annual_manpower_impact_fte
  - Annual Benefit: use annual_benefit
- Do not recompute or alter the computed values.
- If a computed value is null, state that it could not be calculated because the required information was not found.

"""

# =========================================================
# MAIN ENDPOINT
# =========================================================

@app.post("/process")
async def process(query: str = Form(""), file: UploadFile = File(None)):

    if file:
        if not file.filename.lower().endswith(".docx"):
            raise HTTPException(
                status_code=400,
                detail="Please upload your AOR as a Word document (.docx) file."
            )

        contents = await file.read()

        raw_docs = extract_documents(contents, file.filename)

        if not raw_docs:
            raise HTTPException(
                status_code=400,
                detail="The document appears to be empty or unreadable."
            )
   
    else:
        if not query.strip():
            raise HTTPException(
                 status_code=400,
                 detail="Please provide either a file or some text."
            )

        raw_docs = [
            LCDocument(
                page_content=query,
                metadata={
                    "filename":"user_prompt",
                    "source_type":"prompt",
                    "content_type":"plain_text"
                }
            )
        ]

    embeddings = OpenAIEmbeddings()
    docs = split_documents(raw_docs, embeddings)

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No readable content could be extracted from the document."
        )

    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 8})

    retrieval_query = query.strip() or (
        "project objective business need benefits "
        "cost estimate capex opex manpower impact "
        "fte time saving annual benefit pv factor "
        "implementation period project duration"
    )

    relevant_docs = retriever.invoke(retrieval_query)

    context_blocks = []
    for i, doc in enumerate(relevant_docs, start=1):
        metadata = doc.metadata
        source_parts = []
        if metadata.get("filename"):
            source_parts.append(f"File: {metadata.get('filename')}")
        if metadata.get("page"):
            source_parts.append(f"Page: {metadata.get('page')}")
        if metadata.get("sheet"):
            source_parts.append(f"Sheet: {metadata.get('sheet')}")
        if metadata.get("table_index"):
            source_parts.append(f"Table: {metadata.get('table_index')}")
        if metadata.get("row_index"):
            source_parts.append(f"Row: {metadata.get('row_index')}")
        if metadata.get("chunk_index"):
            source_parts.append(f"Chunk: {metadata.get('chunk_index')}")
        source_label = " | ".join(source_parts)
        context_blocks.append(f"[Chunk {i} | {source_label}]\n{doc.page_content}")

    context = "\n\n".join(context_blocks)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extraction_prompt = build_extraction_prompt(context)
    extraction_response = llm.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content="Extract ICT AOR inputs from the context.")
    ])

    try:
        extracted_inputs_raw = parse_json_from_llm(extraction_response.content)
        extracted_inputs = clean_extracted_inputs(extracted_inputs_raw)
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to extract structured ICT AOR inputs.",
            "details": str(e),
            "raw_llm_response": extraction_response.content
        }

    computed_metrics = compute_aor_metrics(extracted_inputs)
    approval_info = determine_approving_authority(extracted_inputs.get("grand_total"))
    missing_fields = identify_missing_fields(extracted_inputs, computed_metrics)
    submission_complete = len(missing_fields) == 0

    aor_prompt = build_aor_drafting_prompt(
        context=context,
        inputs=extracted_inputs,
        metrics=computed_metrics,
        approval_info=approval_info
    )

    final_response = llm.invoke([
        SystemMessage(content=aor_prompt),
        HumanMessage(content="Draft the AOR.")
    ])

    full_result = (
        final_response.content.rstrip()
        + "\n\n"
        + build_missing_information_section(missing_fields)
    )

    missing_sections = []
    if file and file.filename.lower().endswith(".docx"):
        try:
            candidate_headings = extract_candidate_headings(contents)
            structure_prompt = build_structure_review_prompt(candidate_headings, STANDARD_AOR_SECTIONS)
            structure_response = llm.invoke([
                SystemMessage(content=structure_prompt),
                HumanMessage(content="Assess the document's structure against the standard AOR format.")
            ])
            structure_data = parse_json_from_llm(structure_response.content)
            if isinstance(structure_data, dict):
                missing_sections = structure_data.get("missing_sections") or []
                order_note = structure_data.get("order_note") or ""
            else:
                order_note = ""
            full_result += "\n\n" + build_structure_review_section(missing_sections, order_note)
        except Exception:
            pass

    amended_docx_base64 = None
    amended_docx_filename = None

    import logging
    logging.warning(f"EXTRACTED INPUTS: {json.dumps(extracted_inputs, indent=2)}")
    logging.warning(f"COMPUTED METRICS: {json.dumps(computed_metrics, indent=2)}")

    if file and (missing_fields or missing_sections):
        amended_docx_base64 = base64.b64encode(
            build_amended_docx(contents, missing_fields, missing_sections)
        ).decode("ascii")
        amended_docx_filename = f"amended-{file.filename}"
    elif not file:
        amended_docx_base64 = base64.b64encode(
            build_result_docx(full_result, inputs=extracted_inputs, metrics=computed_metrics)
        ).decode("ascii")
        amended_docx_filename = "aor-assessment.docx"

    return {
        "status": "ok",
        "approval_info": approval_info,
        "submission_complete": submission_complete,
        "result": full_result,
        "extracted_inputs": extracted_inputs,
        "computed_metrics": computed_metrics,
        "missing_fields": missing_fields,
        "amended_docx_base64": amended_docx_base64,
        "amended_docx_filename": amended_docx_filename,
        "sources_used": [
            {
                "metadata": doc.metadata,
                "preview": doc.page_content[:300]
            }
            for doc in relevant_docs
        ]
    }