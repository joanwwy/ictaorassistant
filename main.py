from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import os

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

    # Query the chunks
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=query)

    return { "result": result }from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import os

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

    # Query the chunks
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=query)

    return { "result": result }