import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    #query_text = """We are looking for a Golang Developer with 0–2 years of experience, a strong networking background, and hands-on experience in drone development to build and maintain scalable backend systems, integrate communication protocols for UAVs, and collaborate on real-time drone control and telemetry solutions."
    query_text = """
Job Description:
FASTAPI required
B Tech required
Drone Development required 
Python required
"""

    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200).to(device)
    formatted_response  = tokenizer.decode(output[0], skip_special_tokens=True)
    # model = AzureChatOpenAI(model_name="gpt-4.1", temperature=0,api_key=os.getenv("AZURE_OPENAI_API_KEY"),api_version="2025-01-01-preview")
    # response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return output


if __name__ == "__main__":
    main()
