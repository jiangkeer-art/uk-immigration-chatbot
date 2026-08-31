import os
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "./immigration_data"
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8001
COLLECTION_NAME = "langchain"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        embedding_function=embeddings
    )

    return vectordb

def find_latest_file(pattern):
    files = list(Path(DATA_DIR).glob(pattern))
    if not files:
        return None
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]

def load_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text(separator=" ", strip=True)
    return Document(
        page_content=text,
        metadata={"source": str(filepath), "source_type": "html"}
    )

def load_pdf(filepath):
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(filepath)
        doc.metadata["source_type"] = "pdf"
    return docs

def main():

    if not Path(DATA_DIR).exists():
        return

    files = list(Path(DATA_DIR).glob("*"))
    print(f"Files: {len(files)}")
    for f in files:
        print(f"   - {f.name} ({f.stat().st_size} bytes)")

    html_pattern = "statement_of_change_HC*.html"
    pdf_pattern = "immigration_rules_full_*.pdf"

    html_files = list(Path(DATA_DIR).glob(html_pattern))
    pdf_files = list(Path(DATA_DIR).glob(pdf_pattern))


    html_file = sorted(html_files, key=lambda f: f.stat().st_mtime)[-1] if html_files else None
    pdf_file = sorted(pdf_files, key=lambda f: f.stat().st_mtime)[-1] if pdf_files else None

    print(f"new HTML: {html_file.name if html_file else 'null'}")
    print(f"new PDF : {pdf_file.name if pdf_file else 'null'}")

    all_docs = []

    if html_file:
        try:
            doc = load_html(html_file)
            if doc and doc.page_content:
                all_docs.append(doc)
            else:
                print("HTML null")
        except Exception as e:
            print(f"HTML load failed: {e}")
            import traceback
            traceback.print_exc()

    if pdf_file:
        try:
            docs = load_pdf(pdf_file)
            if docs:
                all_docs.extend(docs)
            else:
                print("PDF null")
        except Exception as e:
            print(f"PDF load failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)


    try:

        vectordb = get_vectordb()

        old_data = vectordb.get()
        old_ids = old_data.get("ids", [])

        if old_ids:
            vectordb.delete(ids=old_ids)

        vectordb.add_documents(chunks)
        count = vectordb._collection.count()

    except Exception as e:
        print(f"database update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()