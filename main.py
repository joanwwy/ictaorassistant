from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

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
    # 1. Read and decode the uploaded file
    contents = await file.read()
    text = contents.decode("utf-8")

    # 2. Initialize Embeddings and Semantic Chunker
    embeddings = OpenAIEmbeddings()
    splitter = SemanticChunker(embeddings)
    docs = splitter.create_documents([text])

    # 3. Create a temporary in-memory Vector Store from the semantic chunks
    # (Using FAISS requires installing `pip install faiss-cpu`)
    vector_store = FAISS.from_documents(docs, embeddings)

    # 4. Retrieve only the top relevant chunks (e.g., top 4) matching the query
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    
    # 5. Combine only the relevant chunks into the context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 6. Generate the answer using ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini") # Specified a cost-effective model
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Use the following retrieved pieces of context "
            "to answer the user's question. If you don't know the answer, just say that "
            "you don't know.\n\n"
            f"Context:\n{context}"
        )),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)

    return {"result": response.content}