import os
import shutil
import time
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
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
PERSIST_DIR = "./immigration_db2"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def smart_extractor(html: str) -> str:

    soup = BeautifulSoup(html, "html.parser")

    # 语言检测
    html_tag = soup.find("html")
    lang = html_tag.get("lang") if html_tag else None

    if lang and not lang.startswith("en"):
        return ""

    # 清理噪音
    for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if not main:
        return ""

    # 提取标题
    title_tag = main.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # 提取正文
    body_text = main.get_text(separator=" ", strip=True)

    if len(body_text) < 50:
        print("    [跳过] main 内内容过短")
        return ""

    #lang 属性缺失
    if not lang or not lang.startswith("en"):
        try:
            detected = detect(body_text)
            if detected != "en":
                return ""
        except Exception:
            pass

    print(f"提取标题: {title[:40]}... 内容长度: {len(body_text)} 字符")
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

            # 打印每个成功提取的文档
            print(f"成功提取 {len(docs)} 个页面")
            for doc in docs:
                source_url = doc.metadata.get('source', '')
                all_urls.append(source_url)
                print(f" {source_url}")

            all_documents.extend(docs)

        except Exception as e:
            print(f"抓取失败: {e}")
            failed_urls.append(start_url)

        time.sleep(2)

    # 汇总
    print("\n" + "-" * 70)
    print(f"总共抓取到 {len(all_documents)} 个页面")

    if all_urls:
        with open("captured_urls.txt", "w", encoding="utf-8") as f:
            for url in all_urls:
                f.write(url + "\n")
        print(f"所有 URL 已保存到 captured_urls.txt（共 {len(all_urls)} 个）")
        print("\n前 5 个示例:")
        for idx, url in enumerate(all_urls[:5], 1):
            print(f"  {idx}. {url}")
        if len(all_urls) > 5:
            print(f"  ... 还有 {len(all_urls)-5} 个，详见文件")
    else:
        print("没有提取到任何页面")

    if failed_urls:
        print(f"\n以下入口抓取失败:")
        for url in failed_urls:
            print(f"  - {url}")

    if not all_documents:
        print("\n没有有效文档，程序终止。")
        return

    # 分块
    print("\n正在分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"生成 {len(chunks)} 个文本块")

    print("\n构建向量数据库...")
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("已删除旧数据库")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"加载模型: {EMBEDDING_MODEL}")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print(f"数据库保存至: {PERSIST_DIR}")
    except Exception as e:
        print(f"构建失败: {e}")
        return

    print("完成")
    print(f"原始页面数: {len(all_documents)}")
    print(f"文本块: {len(chunks)}")



if __name__ == "__main__":
    main()