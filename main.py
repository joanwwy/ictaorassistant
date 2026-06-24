from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
    text = contents.decode("utf-8")

    splitter = SemanticChunker(OpenAIEmbeddings())
    docs = splitter.create_documents([text])

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the following document to answer the user's question.\n\n" + context),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)

    return {"result": response.content}