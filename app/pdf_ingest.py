import os
from agno.document.base import Document

from agno.document.chunking.document import DocumentChunking
from agno.knowledge.text import TextKnowledgeBase, TextReader
from agno.vectordb.milvus.milvus import Milvus
from agno.embedder.google import GeminiEmbedder

from app.process import ExtractedPDF, process_data
from uuid import uuid4

MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_TOKEN = os.environ["MILVUS_TOKEN"]
LLM_API_KEY = os.environ["LLM_API_KEY"]


knowledge_base = TextKnowledgeBase(
    path="data/pdfs",
    vector_db=Milvus(
        collection="openspecimen_confluence",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        embedder=GeminiEmbedder(api_key=LLM_API_KEY),
    ),
    reader=TextReader(chunk=True, chunking_strategy=DocumentChunking()),
)


def add_new_document(doc_text: str):
    knowledge_base.load_text(doc_text, upsert=True)


def add_documents(docs: list[Document]):
    knowledge_base.load_documents(docs, upsert=True)


def add_document(docs: list[Document]):
    knowledge_base.load_document(docs, upsert=True)


if __name__ == "__main__":
    extracted_pdfs: dict[str, ExtractedPDF] = process_data()

    documents = []
    for k, pdf in extracted_pdfs.items():
        for page in pdf.pages:
            documents.append(
                Document(
                    content=page.markdown,
                    id=f"{pdf.file_name}_{uuid4()}",
                    meta_data=page.metadata,
                )
            )
            # try:
            #     json.dumps(page.metadata)
            # except Exception as err:
            #     print(err)
            #     print(f"{page.metadata=}")
    count = 0
    failed = 0
    for doc in documents:
        try:
            add_document(doc)
            count += 1
        except Exception as err:
            print(err)
            print(f"{doc=}")
            failed += 1
    print(f"Total documents: {len(documents)}")
    print(f"Successfully added: {count}")
    print(f"Failed to add: {failed}")
