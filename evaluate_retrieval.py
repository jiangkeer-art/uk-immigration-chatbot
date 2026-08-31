import csv
import hashlib

from rag_module import (
    vectordb,
    vectordb2,
    bm25_search,
    rebuild_bm25,
    rewrite_query_multi,
    reranker
)



QUESTIONS = [
    (
        "Q01",
        "My wife is a British citizen and I want to move to the UK "
        "to live with her. Which visa should I apply for?"
    ),
    (
        "Q02",
        "What is the minimum income requirement for a partner or "
        "spouse family visa?"
    ),
    (
        "Q03",
        "What evidence can I provide to prove that my relationship "
        "with my partner is genuine?"
    ),
    (
        "Q04",
        "I want to come to the UK to marry my British partner. "
        "How long can I stay as a fiancé or fiancée, and what "
        "happens after we marry?"
    ),
    (
        "Q05",
        "How much money do I need to show when applying for a UK "
        "Student visa?"
    ),
    (
        "Q06",
        "What documents do I need to provide when applying for a "
        "Student visa?"
    ),
    (
        "Q07",
        "Do all Student visa applicants need to provide financial "
        "evidence?"
    ),
    (
        "Q08",
        "I have already paid part of my university tuition fee. "
        "How does this affect the amount of money I need to show "
        "for my Student visa?"
    ),
    (
        "Q09",
        "What salary do I normally need to qualify for a Skilled "
        "Worker visa?"
    ),
    (
        "Q10",
        "Does any job offer from a UK company allow me to apply "
        "for a Skilled Worker visa?"
    ),
    (
        "Q11",
        "How much money do I need to pay and have available when "
        "applying for a Skilled Worker visa?"
    ),
    (
        "Q12",
        "My occupation is on the Immigration Salary List. Does "
        "that change the salary requirement for my Skilled Worker visa?"
    ),
    (
        "Q13",
        "How long can I normally stay in the UK with a Standard "
        "Visitor visa?"
    ),
    (
        "Q14",
        "If I have a 5-year Standard Visitor visa, can I stay in "
        "the UK continuously for five years?"
    ),
    (
        "Q15",
        "How early can I apply for a Standard Visitor visa before "
        "How early can I apply for a Standard Visitor visa before "
        "travelling to the UK?"
    ),
    (
        "Q16",
        "After living in the UK on a Skilled Worker visa, when can "
        "I apply for indefinite leave to remain?"
    ),
    (
        "Q17",
        "Do I still need to meet a salary requirement when applying "
        "for indefinite leave to remain as a Skilled Worker?"
    ),
    (
        "Q18",
        "I have spent some time outside the UK while holding a "
        "Skilled Worker visa. Could this affect my application "
        "for settlement?"
    ),
    (
        "Q19",
        "I already have a Skilled Worker visa but want to change "
        "to a different employer. Do I need to do anything to my visa?"
    ),
    (
        "Q20",
        "I am currently studying in the UK and have received a job "
        "offer. Can I switch from a Student visa to a Skilled Worker visa?"
    ),
    (
        "Q21",
        "What documents and information do I need when applying "
        "to visit the UK as a tourist?"
    ),
    (
        "Q22",
        "My partner lives in Britain and I want to stay with them "
        "permanently. What route should I look at?"
    ),
    (
        "Q23",
        "I want to visit my family in Britain for about eight months. "
        "Can I just use a normal visitor visa?"
    ),
    (
        "Q24",
        "I have been offered a job in the UK with a salary of "
        "£40,000 per year. Does that mean I qualify for a Skilled "
        "Worker visa?"
    ),
    (
        "Q25",
        "I have £12,000 in savings and want to apply for a Student "
        "visa. Is that enough?"
    ),
]


def make_passage_id(text):

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()[:12]


def retrieve_bm25(question, k=4):

    results = bm25_search(question, k=k)

    final_results = []

    for doc, score in results:
        final_results.append({
            "document": doc,
            "score": float(score)
        })

    return final_results


def retrieve_dense(question, k=4):

    results_db1 = vectordb.similarity_search_with_score(
        question,
        k=k
    )

    results_db2 = vectordb2.similarity_search_with_score(
        question,
        k=k
    )

    combined_results = results_db1 + results_db2

    combined_results.sort(
        key=lambda item: item[1]
    )

    seen_contents = set()
    final_results = []

    for doc, score in combined_results:

        content = doc.page_content

        if content in seen_contents:
            continue

        seen_contents.add(content)

        final_results.append({
            "document": doc,
            "score": float(score)
        })

        if len(final_results) >= k:
            break

    return final_results



def retrieve_full(question, k=4):

    search_queries = rewrite_query_multi(
        question,
        num_queries=3
    )

    seen_contents = set()
    all_docs = []

    for query in search_queries:


        docs1 = vectordb.similarity_search(
            query,
            k=3
        )

        docs2 = vectordb2.similarity_search(
            query,
            k=3
        )

        for doc in docs1 + docs2:

            content = doc.page_content

            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(doc)



        bm25_results = bm25_search(
            query,
            k=3
        )

        for doc, score in bm25_results:

            content = doc.page_content

            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(doc)

    if not all_docs:
        return []

    pairs = [
        [question, doc.page_content]
        for doc in all_docs
    ]

    scores = reranker.predict(pairs)

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    final_results = []

    for i in sorted_indices[:k]:

        final_results.append({
            "document": all_docs[i],
            "score": float(scores[i])
        })

    return final_results



def run_evaluation():
    rebuild_bm25()

    methods = {
        "BM25": retrieve_bm25,
        "Dense": retrieve_dense,
        "Full": retrieve_full
    }

    rows = []

    total_questions = len(QUESTIONS)

    for question_number, (qid, question) in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"Question {question_number}/{total_questions}: {qid}"
        )
        print(question)


        for method_name, retrieval_function in methods.items():

            print(f"\nRunning: {method_name}")

            try:

                results = retrieval_function(
                    question,
                    k=4
                )

                print(
                    f"Returned {len(results)} passages."
                )

                for rank, result in enumerate(
                    results,
                    start=1
                ):

                    doc = result["document"]
                    score = result["score"]

                    passage = doc.page_content
                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    pid = make_passage_id(
                        passage
                    )

                    rows.append({
                        "question_id": qid,
                        "question": question,
                        "method": method_name,
                        "rank": rank,
                        "passage_id": pid,
                        "score": score,
                        "source": source,
                        "passage": passage,
                        "relevant": ""
                    })

                    print(
                        f"  Rank {rank} "
                        f"| Passage {pid} "
                        f"| Score: {score:.4f}"
                    )

            except Exception as e:

                print(
                    f"Error while running "
                    f"{method_name} for {qid}: {e}"
                )



    output_file = "retrieval_evaluation.csv"

    fieldnames = [
        "question_id",
        "question",
        "method",
        "rank",
        "passage_id",
        "score",
        "source",
        "passage",
        "relevant"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_evaluation()