from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import Chroma
import torch

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define canonical sections
canonical_sections = [
    "contact", "summary", "skills", "experience", "education",
    "certifications", "projects", "achievements"
]
section_embeddings = model.encode(canonical_sections, convert_to_tensor=True)

# Section prediction function
def predict_section(text_chunk: str, threshold: float = 0.4) -> str:
    text_embedding = model.encode(text_chunk, convert_to_tensor=True)
    print(text_embedding)
    cosine_scores = util.pytorch_cos_sim(text_embedding, section_embeddings)[0]
    best_score, best_index = torch.max(cosine_scores, dim=0)
    best_section = canonical_sections[best_index]
    return best_section if best_score >= threshold else "unknown"


def classify_chroma_chunks(chroma_path: str):
    from langchain.schema import Document

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print(db)
    all_docs = db.get(include=["documents", "metadatas"])

    contents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    ids = all_docs["ids"]

    updated_chunks = []
    for content, metadata, doc_id in zip(contents, metadatas, ids):
        # Ensure the content is a string
        print(content)
        if isinstance(content, list):
            content = "\n".join(content)
        elif not isinstance(content, str):
            content = str(content)

        predicted_section = predict_section(content)
        metadata["predicted_section"] = predicted_section

        doc = Document(page_content=content, metadata=metadata)
        updated_chunks.append((doc, doc_id))
        print(updated_chunks)

    return updated_chunks

def main():
    print(classify_chroma_chunks(CHROMA_PATH))