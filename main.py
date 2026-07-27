import io
import json
import re
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS

from pypdf import PdfReader
from docx import Document as DocxDocument


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


def extract_docx(contents: bytes, filename: str):
    docs = []
    try:
        doc = DocxDocument(io.BytesIO(contents))
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
        if doc.metadata.get("content_type") == "row":
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
        "man_hour_rate"
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

def identify_missing_fields(inputs: dict):
    required_fields = [
        "capex",
        "opex",
        "project_duration_years",
        "annual_productivity_time_savings_hours",
        "annual_manpower_impact_fte",
        "annual_benefit"
    ]
    missing = []
    for field in required_fields:
        value = inputs.get(field)
        if value is None:
            missing.append(field)
        elif isinstance(value, str) and value.strip() == "":
            missing.append(field)
        elif isinstance(value, list) and len(value) == 0:
            missing.append(field)
    return missing


# =========================================================
# PROMPTS
# =========================================================

def build_extraction_prompt(context: str):
    return f"""
You are analysing an ICT Approval of Requirements (AOR) submission using the CAAS AOR template.

Your task is to extract RAW INPUTS only.
Do NOT perform calculations.
Do NOT infer missing values.
Do NOT invent figures, benefits, assumptions, or dates.
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
  "key_assumptions": [],
  "source_evidence": []
}}

Field guidance:
- "project_title": title or name of the ICT project.
- "purpose": what the ICT project seeks to achieve.
- "problem_statement": the gap, pain point, or problem being addressed.
- "capex": one-off capital expenditure (implementation, hardware, cybersecurity, contingency).
- "opex": total operating expenditure over the project duration (licences, maintenance, cloud support).
- "project_duration_years": project duration in years (typically 3 to 5 years).
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
- Whether the submission appears complete for AOR review

5. Missing Information / Follow-up Required
- List any missing inputs that prevent assessment or computation.
"""


# =========================================================
# MAIN ENDPOINT
# =========================================================

@app.post("/process")
async def process(query: str = Form(""), file: UploadFile = File(...)):
    contents = await file.read()

    raw_docs = extract_documents(contents, file.filename)

    if not raw_docs:
        raise HTTPException(
            status_code=400,
            detail="The document appears to be empty or unreadable."
        )

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
    missing_fields = identify_missing_fields(extracted_inputs)
    submission_complete = len(missing_fields) == 0

    explanation_prompt = build_explanation_prompt(
        context=context,
        inputs=extracted_inputs,
        metrics=computed_metrics,
        missing_fields=missing_fields
    )

    final_response = llm.invoke([
        SystemMessage(content=explanation_prompt),
        HumanMessage(content=query.strip() or "Provide a full ICT AOR assessment.")
    ])

    return {
        "status": "ok",
        "submission_complete": submission_complete,
        "result": final_response.content,
        "extracted_inputs": extracted_inputs,
        "computed_metrics": computed_metrics,
        "missing_fields": missing_fields,
        "sources_used": [
            {
                "metadata": doc.metadata,
                "preview": doc.page_content[:300]
            }
            for doc in relevant_docs
        ]
    }