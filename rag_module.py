import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import time

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
            model="deepseek-v4-flash",
            temperature=0.3,
            max_tokens=200,
            timeout=15.0
        )
        content = response.choices[0].message.content.strip()
        # 解析成列表，过滤空行和可能的序号
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

# 加载向量数据库（每次启动时加载）
vectordb = Chroma(
    persist_directory="./immigration_db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

vectordb2 = Chroma(
    persist_directory="./immigration_db2",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

def rag_answer(question, debug=False):
    print(f"vectordb 文档数: {vectordb._collection.count()}")
    print(f"vectordb2 文档数: {vectordb2._collection.count()}")

    search_queries = rewrite_query_multi(question, num_queries=3)

    start = time.time()
    seen_contents = set()
    seen_sources = set()
    all_docs = []

    for query in search_queries:
        docs1 = vectordb.similarity_search(query, k=3)
        docs2 = vectordb2.similarity_search(query, k=3)
        for doc in docs1 + docs2:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(doc)
                src = doc.metadata.get("source", "Unknown")
                seen_sources.add(src)

    search_time = time.time() - start
    print(f"[搜索耗时] {search_time:.3f} 秒")

    context = "\n\n".join([doc.page_content for doc in all_docs])

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
            model="deepseek-v4-flash",
            temperature=0.0,
            max_tokens=1500,
            timeout=30.0
        )
    except Exception as e:
        print(f"API调用异常: {e}")
        return f"Error: {e}", []

    answer = response.choices[0].message.content
    if not answer:
        print("API返回空内容:", response)

    sources = list(seen_sources) if seen_sources else []
    return answer, sources


# 测试
if __name__ == "__main__":
    print("移民咨询问答系统已启动（输入 'exit' 或 'quit' 退出）")
    while True:
        question = input("\n请输入您的问题: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            print("问题不能为空，请重新输入。")
            continue

        try:
            ans, src = rag_answer(question)
            print(f"\n回答: {ans}")
            print(f"来源: {src}")
        except Exception as e:
            print(f"处理问题时出错: {e}")
            print("请稍后重试或联系管理员。")