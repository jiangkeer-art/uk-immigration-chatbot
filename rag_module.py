import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import time
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

def rewrite_query_multi(user_query, num_queries=3):
    prompt = f"""You are a query optimizer for a UK immigration advisory system.
User questions may be colloquial, vague, or multifaceted.
Please rephrase the user's question into {num_queries} queries better suited for semantic retrieval within immigration policy documents.
Requirements:
- Each query should use formal, complete keywords—phrasing that would appear in the documents.
- Aim to cover different aspects or phrasings of the question.
- Output the list of queries directly, one per line, in Chinese or English (depending on the language of the question).
- Do not include numbering, explanations, or any extra text.

用户问题：{user_query}
优化后的查询："""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional assistant for query rewriting."},
                {"role": "user", "content": prompt}
            ],
            model="deepseek-v4-pro",
            temperature=0.3,
            max_tokens=500,
            timeout=15.0,
            extra_body = {"thinking": {"type": "disabled"}}
        )
        content = response.choices[0].message.content.strip()
        lines = content.split('\n')
        rewritten = []
        for line in lines:
            line = line.strip()
            if line:
                if line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[1]
                elif line.startswith('- '):
                    line = line[2:]
                rewritten.append(line)
        if not rewritten:
            return [user_query]
        return rewritten
    except Exception as e:
        print(f"Query rewriting failed; using the original query.: {e}")
        return [user_query]

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma(
    collection_name="langchain",
    host="127.0.0.1",
    port=8000,
    embedding_function=embeddings
)

vectordb2 = Chroma(
    collection_name="langchain",
    host="127.0.0.1",
    port=8001,
    embedding_function=embeddings
)

reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)

def build_bm25(db1, db2):
    unique_docs = []
    seen_contents = set()

    for db in (db1, db2):
        data = db.get(include=["documents", "metadatas"])

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        for i, text in enumerate(documents):
            if not text or text in seen_contents:
                continue

            seen_contents.add(text)

            metadata = {}
            if i < len(metadatas) and metadatas[i]:
                metadata = metadatas[i]

            unique_docs.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

    tokenized = [
        doc.page_content.lower().split()
        for doc in unique_docs
    ]

    bm25 = BM25Okapi(tokenized)

    return bm25, unique_docs

bm25_index, bm25_docs = build_bm25(vectordb, vectordb2)

def rebuild_bm25():
    global bm25_index, bm25_docs
    bm25_index, bm25_docs = build_bm25(
        vectordb,
        vectordb2
    )

def bm25_search(query, k=3):
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)

    top_n = np.argsort(scores)[::-1][:k]

    return [
        (bm25_docs[i], scores[i])
        for i in top_n
    ]


def rag_answer(question, debug=False):
    rebuild_bm25()

    search_queries = rewrite_query_multi(question, num_queries=3)

    start = time.time()
    seen_contents = set()
    all_docs = []

    for query in search_queries:
        docs1 = vectordb.similarity_search(query, k=3)
        docs2 = vectordb2.similarity_search(query, k=3)
        for doc in docs1 + docs2:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(doc)

        bm25_results = bm25_search(query, k=3)

        for doc, score in bm25_results:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(doc)

    search_time = time.time() - start
    print(f"search time{search_time:.3f} 秒")

    pairs = [[question, doc.page_content] for doc in all_docs]
    scores = reranker.predict(pairs)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_docs = [all_docs[i] for i in sorted_indices[:4]]

    context_blocks = []

    for i, doc in enumerate(top_docs, start=1):
        source = doc.metadata.get("source", "Unknown")

        context_blocks.append(
            f"[Source {i}] {source}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a UK immigration advisor. Answer the question based on the provided context.

If the context contains a clear answer, quote it and mention the source.
If the context has conflicting information, list both.
If the context does not provide enough information, politely offer general advice without explicitly stating that you lack information. For example, you can say "For detailed guidance, please check the official UK government website" or suggest contacting a professional.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise UK immigration advisor."},
                {"role": "user", "content": prompt},
            ],
            model="deepseek-v4-pro",
            temperature=0.0,
            max_tokens=2000,
            timeout=30.0,
            extra_body = {"thinking": {"type": "disabled"}}
        )
    except Exception as e:
        return f"Error: {e}", []

    answer = response.choices[0].message.content
    if not answer:
        print("API return null", response)

    sources = []
    seen_final_sources = set()

    for doc in top_docs:
        source = doc.metadata.get("source")

        if (
                source
                and source != "Unknown"
                and source not in seen_final_sources
        ):
            seen_final_sources.add(source)
            sources.append(source)

    return answer, sources