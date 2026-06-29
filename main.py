import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from docx import Document  # Added for Word documents

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
    # 1. Read the uploaded file bytes
    contents = await file.read()
    text = ""

    # 2. Extract text based on file extension
    filename_lower = file.filename.lower()
    
    if filename_lower.endswith(".txt"):
        text = contents.decode("utf-8")
        
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
            
    elif filename_lower.endswith(".docx"):
        try:
            doc_stream = io.BytesIO(contents)
            doc = Document(doc_stream)
            
            # Extract text from paragraphs and table cells
            extracted_paragraphs = [para.text for para in doc.paragraphs if para.text]
            text = "\n".join(extracted_paragraphs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse Word Document: {str(e)}")
            
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a .txt, .pdf, or .docx file."
        )

    # Check if we actually found text to avoid empty document errors
    if not text.strip():
        raise HTTPException(status_code=400, detail="The document appears to be empty or unreadable.")

    # 3. Initialize Embeddings and Semantic Chunker
    embeddings = OpenAIEmbeddings()
    splitter = SemanticChunker(embeddings)
    docs = splitter.create_documents([text])

    # 4. Create a temporary in-memory Vector Store from the semantic chunks
    vector_store = FAISS.from_documents(docs, embeddings)

    # 5. Retrieve only the top relevant chunks matching the query
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    
    # 6. Combine only the relevant chunks into the context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 7. Initialize the Language Model
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 8. Define the prompt
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
    
    # 9. Invoke the model and return response
    response = llm.invoke(messages)
    return {"result": response.content}