import argparse
import os
import shutil
from langchain_community.document_loaders import UnstructuredMarkdownLoader , PyPDFDirectoryLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"
DATA_PATH_Md = "data/resources/"

def main():

	# Check if the database should be cleared (using the --clear flag).
	parser = argparse.ArgumentParser()
	parser.add_argument("--reset", action="store_true", help="Reset the database.")
	args = parser.parse_args()
	if args.reset:
		print("✨ Clearing Database")
		clear_database()

	# Create (or update) the data store.
	documents = load_documents()
	chunks = chunk_documents_by_resume(documents)
	print(f"Number of documents: {len(chunks)}")
	add_to_chroma(chunks)


def load_documents():
	loader = PyPDFDirectoryLoader(DATA_PATH)
	docs = loader.load()
	for doc in docs:
		source_path = doc.metadata.get("source", "")
		resume_id = os.path.basename(source_path)
		doc.metadata["resume_id"] = resume_id
	return docs


from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents_by_resume(documents: list[Document]) -> list[Document]:
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=800,
		chunk_overlap=80,
		length_function=len,
		is_separator_regex=False,
	)

	resume_chunks = []

	for doc in documents:
		resume_id = doc.metadata["resume_id"]
		splits = text_splitter.split_text(doc.page_content)

		for idx, chunk_text in enumerate(splits):
			chunk = Document(
				page_content=chunk_text,
				metadata={
					"resume_id": resume_id,
					"id": f"{resume_id}:{idx}",
					**doc.metadata,  # preserve original metadata
				},
			)
			resume_chunks.append(chunk)

	return resume_chunks



def add_to_chroma(chunks: list[Document]):
	# Load the existing database.
	model = get_embedding_function()
	#embeddings = model.encode(chunks, show_progress_bar=True).tolist()
	db = Chroma(
		persist_directory=CHROMA_PATH, embedding_function=model
	)

	# Calculate Page IDs.
	chunks_with_ids = calculate_chunk_ids(chunks)

	# Add or Update the documents.
	existing_items = db.get(include=[])  # IDs are always included by default
	existing_ids = set(existing_items["ids"])
	print(f"Number of existing documents in DB: {len(existing_ids)}")

	# Only add documents that don't exist in the DB.
	new_chunks = []
	for chunk in chunks_with_ids:
		if chunk.metadata["id"] not in existing_ids:
			new_chunks.append(chunk)

	if len(new_chunks):
		print(f"👉 Adding new documents: {len(new_chunks)}")
		new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
		db.add_documents(new_chunks, ids=new_chunk_ids)
		db.persist()
	else:
		print("✅ No new documents to add")


def calculate_chunk_ids(chunks: list[Document]):
	last_page_id = None
	current_chunk_index = 0

	for chunk in chunks:
		source = chunk.metadata.get("source")
		page = chunk.metadata.get("page", 0)
		resume_id = os.path.basename(source)
		chunk.metadata["resume_id"] = resume_id

		current_page_id = f"{source}:{page}"
		if current_page_id == last_page_id:
			current_chunk_index += 1
		else:
			current_chunk_index = 0

		chunk_id = f"{current_page_id}:{current_chunk_index}"
		chunk.metadata["id"] = chunk_id
		last_page_id = current_page_id

	return chunks



def clear_database():
	if os.path.exists(CHROMA_PATH):
		shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
	main()
