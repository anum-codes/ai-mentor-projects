from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# FIXED: Switched to the active, supported mainline model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def load_and_index_pdf(pdf_path):
    print(f"📄 Loading: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")

    print("🔢 Creating embeddings (this takes ~30 seconds)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector database created")
    return vectorstore

def format_chat_history(chat_history):
    """Formats list of tuples into a string block for the prompt"""
    formatted_history = ""
    for human, ai in chat_history:
        formatted_history += f"Human: {human}\nAI: {ai}\n"
    return formatted_history if formatted_history else "No previous history."

def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def main():
    if len(sys.argv) < 2:
        print("Usage: python day8.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    vectorstore = load_and_index_pdf(pdf_path)
    chain, retriever = create_rag_chain(vectorstore)

    print("\n🤖 RAG system ready! Ask questions about your document.")
    print("Type 'quit' to exit\n")
    print("=" * 50)

    chat_history = []

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        if not question:
            continue

        print("🔍 Searching relevant chunks...")

        # 1. Fetch relevant source documents for page reference printing
        source_docs = retriever.invoke(question)

        # 2. Format the local history array into a readable string
        history_str = format_chat_history(chat_history)

        # 3. Invoke chain with a unified input dictionary
        answer = chain.invoke({
            "question": question, 
            "chat_history": history_str
        })

        print(f"\n📝 Answer:\n{answer}")

        # Show source pages
        pages = set()
        for doc in source_docs:
            if 'page' in doc.metadata:
                pages.add(doc.metadata['page'] + 1)
        if pages:
            print(f"\n📌 Sources: Pages {sorted(pages)}")

        # Save this exchange to the conversation history
        chat_history.append((question, answer))
        print("-" * 50)

if __name__ == "__main__":
    main()