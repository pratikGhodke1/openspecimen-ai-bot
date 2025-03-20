import json
import os
from typing import List
from agno.document.base import Document

# from agno.document.chunking.recursive import RecursiveChunking
from agno.document.chunking.document import DocumentChunking
from agno.knowledge.text import TextKnowledgeBase, TextReader
from agno.vectordb.milvus.milvus import Milvus
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder

from app.process import ExtractedPDF, process_data
from uuid import uuid4

MILVUS_URI = os.environ["MILVUS_URI"]
MILVUS_TOKEN = os.environ["MILVUS_TOKEN"]


knowledge_base = TextKnowledgeBase(
    path="data/pdfs",
    # Table name: ai.pdf_documents
    vector_db=Milvus(
        # collection="pdf_documents_maths",
        # collection="pdf_documents_maths_mod",
        collection="openspecimen_confluence",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        embedder=SentenceTransformerEmbedder(
            id="sentence-transformers/all-mpnet-base-v2", dimensions=768
        ),
        # db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
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
    # print()
    # print()
    # print(extracted_pdfs["iemh1a2.pdf"].pages[0])
    # print()
    # print(extracted_pdfs["iemh1a2.pdf"].pages[1])
    # print()
    # print(extracted_pdfs["iemh1a2.pdf"].pages[2])
    # print()
    # print()
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
