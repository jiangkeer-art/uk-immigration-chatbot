import os
import csv
import time


from rag_module import rag_answer, client

OUTPUT_FILE = "response_evaluation.csv"

MODEL_NAME = "deepseek-v4-pro"



QUESTIONS = [
    (
        "Q01",
        "My wife is a British citizen and I want to move to the UK to live with her. Which visa should I apply for?"
    ),
    (
        "Q02",
        "What is the minimum income requirement for a partner or spouse family visa?"
    ),
    (
        "Q03",
        "What evidence can I provide to prove that my relationship with my partner is genuine?"
    ),
    (
        "Q04",
        "I want to come to the UK to marry my British partner. How long can I stay as a fiancé or fiancée, and what happens after we marry?"
    ),
    (
        "Q05",
        "How much money do I need to show when applying for a UK Student visa?"
    ),
    (
        "Q06",
        "What documents do I need to provide when applying for a Student visa?"
    ),
    (
        "Q07",
        "Do all Student visa applicants need to provide financial evidence?"
    ),
    (
        "Q08",
        "I have already paid part of my university tuition fee. How does this affect the amount of money I need to show for my Student visa?"
    ),
    (
        "Q09",
        "What salary do I normally need to qualify for a Skilled Worker visa?"
    ),
    (
        "Q10",
        "Does any job offer from a UK company allow me to apply for a Skilled Worker visa?"
    ),
    (
        "Q11",
        "How much money do I need to pay and have available when applying for a Skilled Worker visa?"
    ),
    (
        "Q12",
        "My occupation is on the Immigration Salary List. Does that change the salary requirement for my Skilled Worker visa?"
    ),
    (
        "Q13",
        "How long can I normally stay in the UK with a Standard Visitor visa?"
    ),
    (
        "Q14",
        "If I have a 5-year Standard Visitor visa, can I stay in the UK continuously for five years?"
    ),
    (
        "Q15",
        "How early can I apply for a Standard Visitor visa before travelling to the UK?"
    ),
    (
        "Q16",
        "After living in the UK on a Skilled Worker visa, when can I apply for indefinite leave to remain?"
    ),
    (
        "Q17",
        "Do I still need to meet a salary requirement when applying for indefinite leave to remain as a Skilled Worker?"
    ),
    (
        "Q18",
        "I have spent some time outside the UK while holding a Skilled Worker visa. Could this affect my application for settlement?"
    ),
    (
        "Q19",
        "I already have a Skilled Worker visa but want to change to a different employer. Do I need to do anything to my visa?"
    ),
    (
        "Q20",
        "I am currently studying in the UK and have received a job offer. Can I switch from a Student visa to a Skilled Worker visa?"
    ),
    (
        "Q21",
        "What documents and information do I need when applying to visit the UK as a tourist?"
    ),
    (
        "Q22",
        "My partner lives in Britain and I want to stay with them permanently. What route should I look at?"
    ),
    (
        "Q23",
        "I want to visit my family in Britain for about eight months. Can I just use a normal visitor visa?"
    ),
    (
        "Q24",
        "I have been offered a job in the UK with a salary of £40,000 per year. Does that mean I qualify for a Skilled Worker visa?"
    ),
    (
        "Q25",
        "I have £12,000 in savings and want to apply for a Student visa. Is that enough?"
    ),
]



def llm_only_answer(question):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise UK immigration advisor."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=2000,
            timeout=60.0,
            extra_body = {"thinking": {"type": "disabled"}}
        )

        answer = response.choices[0].message.content

        if not answer:
            return "ERROR: Empty response"

        return answer
    except Exception as e:
        print(
            f"LLM-only API error: "
            f"{type(e).__name__}: {repr(e)}"
        )
        return f"ERROR: {type(e).__name__}: {e}"



FIELDNAMES = [
    "question_id",
    "question",
    "system",
    "answer",
    "sources",
    "correctness",
    "relevance",
    "completeness"
]


def load_completed():


    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                question_id = row.get("question_id")
                system = row.get("system")
                answer = row.get("answer", "")

                if (
                    question_id
                    and system
                    and answer
                    and not answer.startswith("ERROR:")
                ):
                    completed.add((question_id, system))

    except Exception as e:
        print(f"Could not read existing CSV: {e}")

    return completed


def append_result(row):

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)



def run_evaluation():

    completed = load_completed()
    for index, (question_id, question) in enumerate(QUESTIONS, start=1):
        key = (question_id, "LLM-only")
        if key not in completed:

            baseline_answer = llm_only_answer(question)
            append_result({
                "question_id": question_id,
                "question": question,
                "system": "LLM-only",
                "answer": baseline_answer,
                "sources": "",
                "correctness": "",
                "relevance": "",
                "completeness": ""
            })

            if not baseline_answer.startswith("ERROR:"):
                completed.add(key)

            time.sleep(1)

        else:
            print("\n Skipping.")



        key = (question_id, "RAG")

        if key not in completed:

            try:
                rag_response, sources = rag_answer(question)

                if not rag_response:
                    rag_response = "ERROR: Empty response"

                if sources:
                    sources_text = " | ".join(
                        str(source) for source in sources
                    )
                else:
                    sources_text = ""

            except Exception as e:
                rag_response = f"ERROR: {e}"
                sources_text = ""

            append_result({
                "question_id": question_id,
                "question": question,
                "system": "RAG",
                "answer": rag_response,
                "sources": sources_text,
                "correctness": "",
                "relevance": "",
                "completeness": ""
            })

            if not rag_response.startswith("ERROR:"):
                completed.add(key)

            time.sleep(1)

        else:
            print("\nSkipping.")


if __name__ == "__main__":
    run_evaluation()