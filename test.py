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



    # 按修改时间排序取最新
    html_file = sorted(html_files, key=lambda f: f.stat().st_mtime)[-1] if html_files else None
    pdf_file = sorted(pdf_files, key=lambda f: f.stat().st_mtime)[-1] if pdf_files else None

    print(f"最新的 HTML: {html_file.name if html_file else '无'}")
    print(f"最新的 PDF : {pdf_file.name if pdf_file else '无'}")

    # 3. 加载文档
    all_docs = []

    if html_file:
        try:
            doc = load_html(html_file)
            if doc and doc.page_content:
                all_docs.append(doc)
                print(f"HTML 加载成功，内容长度: {len(doc.page_content)} 字符")
            else:
                print("HTML 加载后内容为空")
        except Exception as e:
            print(f"HTML 加载失败: {e}")
            import traceback
            traceback.print_exc()

    if pdf_file:
        try:
            docs = load_pdf(pdf_file)
            if docs:
                all_docs.extend(docs)
                print(f"PDF 加载成功，共 {len(docs)} 页")
                total_len = sum(len(d.page_content) for d in docs)
                print(f"   总字符数: {total_len}")
            else:
                print("PDF 加载后返回空列表")
        except Exception as e:
            print(f"PDF 加载失败: {e}")
            import traceback
            traceback.print_exc()

    if not all_docs:
        return

    print(f"总文档数: {len(all_docs)}")

    print("开始分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"生成的文本块数: {len(chunks)}")

    if not chunks:
        # 打印前200字符
        for i, doc in enumerate(all_docs):
            print(f"  Doc {i+1} 预览: {doc.page_content[:200]}...")
        return

    print("更新数据库...")
    try:
        print(f"连接 Chroma Server: {CHROMA_HOST}:{CHROMA_PORT}")

        vectordb = get_vectordb()

        # 获取原有数据
        old_data = vectordb.get()
        old_ids = old_data.get("ids", [])

        print(f"数据库原有文本块: {len(old_ids)}")

        # 删除旧数据
        if old_ids:
            print("正在删除旧数据...")
            vectordb.delete(ids=old_ids)
            print("旧数据删除完成")

        # 写入新数据
        print(f"正在写入 {len(chunks)} 个新文本块...")
        vectordb.add_documents(chunks)

        count = vectordb._collection.count()

        print(f"数据库更新完成，当前向量数: {count}")

    except Exception as e:
        print(f"数据库更新失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()