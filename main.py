import io
import pandas as pd  # Added for Excel handling
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from docx import Document

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

@app.post("/process")
async def process(query: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    text = ""
    filename_lower = file.filename.lower()
    
    # --- TEXT FILE ---
    if filename_lower.endswith(".txt"):
        text = contents.decode("utf-8")
        
    # --- PDF FILE ---
    elif filename_lower.endswith(".pdf"):
        try:
            pdf_stream = io.BytesIO(contents)
            pdf_reader = PdfReader(pdf_stream)
            extracted_pages = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            text = "\n".join(extracted_pages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
            
    # --- WORD DOCUMENT ---
    elif filename_lower.endswith(".docx"):
        try:
            doc_stream = io.BytesIO(contents)
            doc = Document(doc_stream)
            extracted_paragraphs = [para.text for para in doc.paragraphs if para.text]
            text = "\n".join(extracted_paragraphs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse Word Document: {str(e)}")
            
    # --- EXCEL SPREADSHEETS ⬇️ ---
    elif filename_lower.endswith((".xlsx", ".xls")):
        try:
            excel_stream = io.BytesIO(contents)
            # sheet_name=None reads ALL sheets into a dictionary of DataFrames
            excel_sheets = pd.read_excel(excel_stream, sheet_name=None)
            
            sheet_texts = []
            for sheet_name, df in excel_sheets.items():
                if not df.empty:
                    # Convert dataframe to a readable string format (CSV format)
                    sheet_data = df.to_csv(index=False)
                    sheet_texts.append(f"--- Sheet: {sheet_name} ---\n{sheet_data}")
                    
            text = "\n\n".join(sheet_texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse Excel Document: {str(e)}")
            
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload .txt, .pdf, .docx, or .xlsx files."
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="The document appears to be empty or unreadable.")

    # 3. Initialize Embeddings and Semantic Chunker
    embeddings = OpenAIEmbeddings()
    splitter = SemanticChunker(embeddings)
    docs = splitter.create_documents([text])

    # 4. Create Vector Store
    vector_store = FAISS.from_documents(docs, embeddings)

    # 5. Retrieve
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 6. Language Model & Prompt Execution
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    system_prompt = (
        "You are an expert financial analyst. Use the following pieces of context to "
        "answer the user's question. Be concise, use bullet points, and if the answer "
        "cannot be found in the context, say 'I cannot find that in the document.'\n\n"
        f"Context:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    return {"result": response.content}