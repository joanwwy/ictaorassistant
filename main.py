import io
import json
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
    return {"status": "ok"}


# ---------- EXTRACTION FUNCTIONS ----------

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
    return docs


def extract_docx(contents: bytes, filename: str):
    docs = []
    doc = DocxDocument(io.BytesIO(contents))

    # Paragraphs
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if paras:
        docs.append(
            LCDocument(
                page_content="\n".join(paras),
                metadata={
                    "filename": filename,
                    "source_type": "docx",
                    "content_type": "paragraphs"
                }
            )
        )

    # Tables
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

    return docs


def extract_excel(contents: bytes, filename: str):
    docs = []
    excel = pd.read_excel(io.BytesIO(contents), sheet_name=None)

    for sheet_name, df in excel.items():
        if df.empty:
            continue

        # Whole sheet pocket
        docs.append(
            LCDocument(
                page_content=df.to_csv(index=False),
                metadata={
                    "filename": filename,
                    "source_type": "excel",
                    "sheet": sheet_name,
                    "content_type": "sheet_table"
                }
            )
        )

        # Row pockets
        for idx, row in df.iterrows():
            row_dict = row.dropna().to_dict()
            if row_dict:
                docs.append(
                    LCDocument(
                        page_content=json.dumps(row_dict),
                        metadata={
                            "filename": filename,
                            "source_type": "excel",
                            "sheet": sheet_name,
                            "content_type": "row",
                            "row_index": idx + 1
                        }
                    )
                )

    return docs


def extract_documents(contents: bytes, filename: str):
    filename = filename.lower()

    if filename.endswith(".txt"):
        return extract_txt(contents, filename)
    if filename.endswith(".pdf"):
        return extract_pdf(contents, filename)
    if filename.endswith(".docx"):
        return extract_docx(contents, filename)
    if filename.endswith((".xlsx", ".xls")):
        return extract_excel(contents, filename)

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format"
    )


# ---------- SPLITTING ----------

def split_documents(raw_docs, embeddings):
    splitter = SemanticChunker(embeddings)
    result = []

    for doc in raw_docs:
        if doc.metadata.get("content_type") == "row":
            result.append(doc)
        else:
            chunks = splitter.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata]
            )
            for i, chunk in enumerate(chunks, start=1):
                chunk.metadata["chunk_index"] = i
                result.append(chunk)

    return result


# ---------- TASK & PROMPT ----------

def infer_task_type(query: str):
    q = query.lower()
    if any(w in q for w in ["summary", "summarise", "overview"]):
        return "summary"
    if any(w in q for w in ["extract", "list", "identify"]):
        return "extraction"
    if any(w in q for w in ["risk", "issue", "gap"]):
        return "risk"
    if any(w in q for w in ["cost", "amount", "budget"]):
        return "financial"
    return "general"


def build_prompt(task_type: str, context: str):
    return f"""
You must answer ONLY using the context below.
If information is missing, say "Not found in document".

Context:
{context}
"""


# ---------- MAIN ENDPOINT ----------

@app.post("/process")
async def process(query: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()

    raw_docs = extract_documents(contents, file.filename)

    embeddings = OpenAIEmbeddings()
    docs = split_documents(raw_docs, embeddings)

    store = FAISS.from_documents(docs, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 6})

    relevant_docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[Pocket {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    )

    prompt = build_prompt(infer_task_type(query), context)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ])

    return {
        "result": response.content
    }