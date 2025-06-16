import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

load_dotenv()

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an expert recruiter. Given a job description and a candidate’s resume sections, rate the candidate from 0 to 100 based on their fit. Explain the score briefly.

Job Description:
{question}

Candidate Resume:
{context}


Respond as: Score: <number>/100. Explanation: <why>.
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    #query_text = """We are looking for a Golang Developer with 0–2 years of experience, a strong networking background, and hands-on experience in drone development to build and maintain scalable backend systems, integrate communication protocols for UAVs, and collaborate on real-time drone control and telemetry solutions."

    #print(f"Query: {query_text}")
    return query_rag_evaluator(query_text)


def query_rag_evaluator(query_text: str):
    # Set up DB and device
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        #print(response_text)
        match = re.search(r"Score:\s*(\d{1,3})/100\.\s*Explanation:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            score = match.group(1).strip()
            explanation = f"Score:{score}/100 Explanation: {match.group(2).strip()}"
        else:
            score = "0"
            explanation = "Could not parse the LLM response."

        scored_resumes.append((resume_id, "Yes", explanation))
        scores_dict.append((resume_id, int(score)))
        explanations_dict.append((resume_id, explanation))

        #print(resume_id)
        #print(f"Score: {score}\nExplanation: {explanation}\n")

    end_time = time.time()
    # Sort scored_resumes by score in descending order using scores_dict
    score_lookup = dict(scores_dict)
    scored_resumes.sort(key=lambda x: score_lookup.get(x[0], 0), reverse=True)
    #print(f"⏱️ Time taken to score {len(scored_resumes)} resumes: {end_time - start_time:.2f} seconds")

    print(json.dumps(scored_resumes))
    return json.dumps(scored_resumes)


if __name__ == "__main__":
    main()
