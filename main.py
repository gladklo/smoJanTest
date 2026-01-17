log_prompts = False

# Load pdf
from langchain_community.document_loaders import PyPDFLoader

# ------------------------------------------------------------
# Bilder holen
import os
os.makedirs("extracted_images", exist_ok=True)

import fitz  # PyMuPDF
from PIL import Image
import io

file_path = "pdf/nke-10k-2023.pdf"
pdf = fitz.open(file_path)

all_images = []
image_metadata = []

for page_num, page in enumerate(pdf, start=1):
    for img_index, img in enumerate(page.get_images(full=True), start=1):
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        image = Image.open(io.BytesIO(image_bytes))

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 🔹 1. kleine Bilder überspringen (Logos, Icons)
        if image.width < 300 or image.height < 300:
            continue

        # 🔹 2. Bildgröße normalisieren
        image.thumbnail((1024, 1024), Image.LANCZOS)

        image_path = f"extracted_images/page_{page_num}_img_{img_index}.png"
        image.save(image_path, format="PNG", optimize=True)


        all_images.append(image)

        image_metadata.append({
            "page": page_num,
            "image_index": img_index,
            "ext": image_ext,
            "path": image_path  # ✅ DAS FEHLTE
        })

print(f"Extracted {len(all_images)} images from the PDF")

# Bilder zu Text
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

vision_llm = ChatOllama(
    model="qwen3-vl:8b",  # oder anderes Vision-Modell
    temperature=0.0
)

def describe_image(image_path: str) -> str:
    prompt = (
        "Analyze this image carefully.\n"
        "Describe in detail what information it contains.\n"
        "If it shows a chart or table, describe trends, numbers and anything that could be relevant. \n" 
        "Your answer will be embedded in a vector database."
    )

    response = vision_llm.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_path},
            ]
        )
    ])

    return response.content

#text zu langchain doc
from langchain_core.documents import Document

image_docs = []

for meta in image_metadata:
    description = describe_image(meta["path"])

    image_docs.append(
        Document(
            page_content=description,
            metadata={
                "type": "image",
                "page": meta["page"],
                "source": "pdf_image"
            }
        )
    )

# ------------------------------------------------------------
# text holen
#file_path = "exampletext.pdf"
file_path = "pdf/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

#print(f"The document has {len(docs)} pages.")

# ------------------------------------------------------------
# Split pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


# ------------------------------------------------------------
# Embed

# -----
# AI
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# -----

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)

#print(f"Generated vectors of length {len(vector_1)}\n")
#print(vector_1[:10])

# ------------------------------------------------------------
# Store vectors
from langchain_core.vectorstores import InMemoryVectorStore

# In memory option
# For more store possiilitys see https://docs.langchain.com/oss/python/langchain/knowledge-base#in-memory
vector_store = InMemoryVectorStore(embeddings)

#Text Speichern
vector_store.add_documents(documents=all_splits)

#Bilder Speichern
vector_store.add_documents(image_docs)


# ------------------------------------------------------------
# Get correct chunks

prompt = "What finance improvements did Nike have from 2022 to 2023"

# Example
results = vector_store.similarity_search_with_score(prompt, k=5)

# ------------------------------------------------------------
# Print the top-k retrieved chunks with scores
print("\n--- Top Chunks Used for Answer ---\n")
for i, (chunk, score) in enumerate(results, start=1):
    print(f"--- Chunk {i} (Score: {score:.3f}) ---\n")
    print(chunk.page_content.strip())
    print("\n" + "-"*80 + "\n")  # Trennlinie für Übersicht

# ------------------------------------------------------------
# Generate final answer with local LLM (RAG)

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

# Chunks zusammenfügen
def build_simple_context(docs, max_chars: int | None = None) -> str:
    """
    Hängt die Dokumente hintereinander, nummeriert sie und (optional) schneidet
    den Gesamttest nach `max_chars` Zeichen ab.
    """
    parts = []
    for i, (doc, score) in enumerate(docs, start=1):
        header = f"\n--- Chunk {i} (Score: {score:.3f}) ---\n"
        parts.append(header + doc.page_content.strip())

    full_text = "\n".join(parts)

    if max_chars is not None and len(full_text) > max_chars:
        # grobe Abschätzung: 1 Zeichen ≈ 0,25 Token → 20 000 Zeichen ≈ 5 000 Token
        full_text = full_text[:max_chars] + " …(truncated)"

    return full_text

context = build_simple_context(results, max_chars=20_000)

system_prompt = """
You are a highly accurate, fact-based question-answering assistant.

You will be given:
- a user question
- a context containing several numbered chunks of text

Rules:
1. Answer ONLY using information from the provided context.
2. For every factual statement, cite the source explicitly using the chunk number, e.g., (Chunk 2).
3. If the answer cannot be found in the context, reply exactly: "The information is not contained in the provided document."
4. Do not invent or assume any information not present in the context.
5. Keep your answer concise and focused on the question.
6. Get as much Information as you can out of the chunks.
"""


user_prompt = f"""
Question:
{prompt}

Context:
{context}
"""

llm = ChatOllama(
    model="gpt-oss:120b-cloud",   # Cloud‑Variante – kein lokaler Download nötig
    temperature=0.0,
    streaming=False,
)


response = llm.invoke(
    [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
)

print("\n--- Final Answer ---\n")
print(response.content)


# Debug / Logs
if log_prompts:
    import json
    from pathlib import Path

    log_file = Path("debug_and_log/tested_with_json.json")

    log_entry = {
        "question": prompt,
        "answer": response.content
    }

    if log_file.exists():
        with log_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with log_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

