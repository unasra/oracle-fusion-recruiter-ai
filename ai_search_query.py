import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time as time
import json


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

load_dotenv()

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# You are a highly precise and strict technical recruiter. You will be given a query and a candidate's resume.
#
# The query may contain specific skills, tools, or qualifications (e.g., "Golang", "FastAPI", "Drone development"). You must only return "Yes" if the resume contains a relevant mention of the required skills or qualifications.
#
# Do NOT assume similar skills are substitutes (e.g., Python is NOT the same as Golang; Django is NOT a substitute for FastAPI). Only answer "Yes" if the match is explicit.
#
# If the resume is missing any critical keyword or skill, respond with "No".
#
# ---
#
# Query:
# {question}
#
# Candidate Resume:
# {context}
#
# ---
#
# Respond strictly in the format:
# Answer: <Yes or No>
# Explanation: <Short explanation of why Yes or No was given, referencing exact resume matches or lack thereof.>
# """

PROMPT_TEMPLATE1 = """
You are a highly precise and strict technical recruiter AI. You will be given a job requirement query (which may include multiple skills or qualifications) and a candidate's resume.

Your task is to assess if **all** skills mentioned in the query are explicitly covered in the resume.

### Rules:
- The match should be treated **case-insensitively**. ("golang", "GOLANG", "Golang" are the same).
- You may consider **common synonyms or direct equivalents** only if they are well-known (e.g., "LLM" ~ "Large Language Model", "Postgres" ~ "PostgreSQL").
- Do NOT assume a match for unrelated technologies. Be strict.
- You MUST return a **single unified response**, using exactly one `Answer:` and one `Explanation:`.
- If even one skill is missing, the answer is "No".

---

Query:
{question}

Candidate Resume:
{context}

---

Respond strictly in this format:

Answer: <Yes or No>  
Explanation: <Short justification including which skills matched (case-insensitive or synonym), and which were missing.>
"""

PROMPT_TEMPLATE2 = """
You are an expert recruiter. Given a job description and a candidate’s resume sections, rate the candidate from 0 to 100 based on their fit. Explain the score briefly.

Job Description:
{question}

Candidate Resume:
{context}


Respond as: Score: <number>/100. Explanation: <why>.
"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    return query_rag_all_resumes(query_text)


def query_rag_all_resumes(query_text: str):
    # Set up DB and device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix: use .startswith and .strip, not .startsWith/.trim
    if query_text.strip().startswith("QUERY:"):
        PROMPT_TEMPLATE = PROMPT_TEMPLATE2
        query_text = query_text.replace("QUERY:", "", 1).strip()
    else:
        PROMPT_TEMPLATE = PROMPT_TEMPLATE1

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Step 1: Fetch all documents in the DB
    results = db.get(include=["documents", "metadatas"])
    all_docs = results["documents"]
    all_metas = results["metadatas"]

    # Step 2: Group documents by resume_id
    resumes = {}
    for doc, meta in zip(all_docs, all_metas):
        resume_id = meta.get("resume_id", "unknown_resume")
        if resume_id not in resumes:
            resumes[resume_id] = []
        resumes[resume_id].append(doc)

    # Step 3: Score each resume
    #print("🎯 Scoring Resumes Based on JD...\n")
    start_time = time.time()
    scored_resumes = []
    scores_dict = []
    explanations_dict = []
    score = "0"

    for resume_id, chunks in resumes.items():
        context_text = "\n\n---\n\n".join(chunks)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text,
            question=query_text.strip()
        )

        # Tokenize and send to LLM
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=200)
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        import re

        # Querying Scores
        match_score = re.search(r"Score:\s*(\d{1,3})/100\.\s*Explanation:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if match_score:
            score = match_score.group(1).strip()
            if int(score) > 50:
                answer = "Yes"
            else:
                answer = "No"
            explanation = f"Score:{score}/100 Explanation: {match_score.group(2).strip()}"
        else:
            match_answer = re.search(r"Answer:\s*(Yes|No)[^\w]*\s*Explanation:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
            if match_answer:
                answer = match_answer.group(1).strip().capitalize()
                explanation = match_answer.group(2).strip()
            else:
                answer = "Unknown"
                score = "0"
                explanation = "Could not parse the LLM response."

        scored_resumes.append((resume_id, answer, explanation))
        scores_dict.append((resume_id, int(score) if score.isdigit() else 0))
        explanations_dict.append((resume_id, explanation))

        #print(resume_id)
        #print(f"Answer: {answer}\nExplanation: {explanation}\n")

    end_time = time.time()
    #print(f"⏱️ Time taken to score {len(scored_resumes)} resumes: {end_time - start_time:.2f} seconds")
    #print(scored_resumes)
    score_lookup = dict(scores_dict)
    scored_resumes.sort(key=lambda x: score_lookup.get(x[0], 0), reverse=True)
    print(json.dumps(scored_resumes))
    return scored_resumes



if __name__ == "__main__":
        main()
