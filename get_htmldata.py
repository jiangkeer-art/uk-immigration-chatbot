import time
import warnings
from bs4 import XMLParsedAsHTMLWarning
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from langdetect import detect

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


START_URLS = [
    "https://www.gov.uk/browse/visas-immigration",
]

MAX_DEPTH = 5
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8000
COLLECTION_NAME = "langchain"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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

def smart_extractor(html: str) -> str:

    soup = BeautifulSoup(html, "html.parser")

    html_tag = soup.find("html")
    lang = html_tag.get("lang") if html_tag else None

    if lang and not lang.startswith("en"):
        return ""

    for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if not main:
        return ""

    title_tag = main.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    body_text = main.get_text(separator=" ", strip=True)

    if len(body_text) < 50:
        return ""

    if not lang or not lang.startswith("en"):
        try:
            detected = detect(body_text)
            if detected != "en":
                return ""
        except Exception:
            pass

    return f"Page Title: {title}\n\nContent: {body_text}"


def main():

    all_documents = []
    all_urls = []
    failed_urls = []

    for start_url in START_URLS:
        print(f"\n正在抓取: {start_url}")

        try:
            loader = RecursiveUrlLoader(
                url=start_url,
                max_depth=MAX_DEPTH,
                extractor=smart_extractor,
                prevent_outside=True,
                headers={"User-Agent": USER_AGENT},
                timeout=20,
                check_response_status=True,
                continue_on_failure=True,
            )

            docs = loader.load()

            for doc in docs:
                source_url = doc.metadata.get('source', '')
                all_urls.append(source_url)
                print(f" {source_url}")

            all_documents.extend(docs)

        except Exception as e:
            failed_urls.append(start_url)

        time.sleep(2)


    if all_urls:
        with open("captured_urls.txt", "w", encoding="utf-8") as f:
            for url in all_urls:
                f.write(url + "\n")
        print(f"all URL saved captured_urls.txt（ {len(all_urls)} ）")
    else:
        print("0")

    if failed_urls:
        print(f"\nfailedurl")
        for url in failed_urls:
            print(f"  - {url}")

    if not all_documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_documents)

    try:
        vectordb = get_vectordb()
        old_data = vectordb.get()
        old_ids = old_data.get("ids", [])
        if old_ids:
            vectordb.delete(ids=old_ids)
        vectordb.add_documents(chunks)
    except Exception as e:
        print(f"failed update{e}")
        import traceback
        traceback.print_exc()
        return




if __name__ == "__main__":
    main()