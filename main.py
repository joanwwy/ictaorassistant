from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process(query: str = Form(...), file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    text = contents.decode("utf-8")

    # Semantic chunking
    splitter = SemanticChunker(OpenAIEmbeddings())
    docs = splitter.create_documents([text])

    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Query using ChatOpenAI directly
    llm = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the following document to answer the user's question.\n\n" + context),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)

    return { "result": response.content }