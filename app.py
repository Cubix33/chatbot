from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


def get_llm():
    # Use Groq instead of OpenAI/Watsonx
    return ChatGroq(
        model="openai/gpt-oss-20b",  # Free model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5
    )


def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    return splitter.split_documents(data)


def vector_database(chunks):
    # Use free embeddings from HuggingFace instead
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(chunks, embeddings)


def retriever(file):
    docs = document_loader(file.name)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    
    prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:
Context: {context}
Question: {question}
Answer:""")
    
    chain = (
        {"context": retriever_obj, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query)


rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF and ask questions. Powered by Groq and LangChain."
)


if __name__ == "__main__":
    rag_application.launch(share=True)
