log_prompts = False

# Load pdf
from langchain_community.document_loaders import PyPDFLoader

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

#print(f"The document has {len(all_splits)} chunks.")

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

ids = vector_store.add_documents(documents=all_splits)

# ------------------------------------------------------------
# Get correct chunks

prompt = "What finance improvements did Nike have from 2023 to 2022"

# Example
results = vector_store.similarity_search_with_score(prompt, k=5)

doc, score = results[0]

for i, (d, s) in enumerate(results, start=1):
    print(f"\n--- Rank {i} ---")
    print(f"Score: {s}")
    print(d.page_content[:300], "...\n\n\n")

print(f"Score: {score}\n")
print(doc.page_content)

# ------------------------------------------------------------
# Generate final answer with local LLM (RAG)

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Local generative LLM
llm = ChatOllama(
    model="llama3",
    temperature=0.0  # deterministic, factual answers
)

# System prompt: instruct the model how to behave
system_prompt = """
You are a factual question-answering assistant.

You will be given:
- a user question
- a context chunk extracted from a document

Rules:
- Answer ONLY using the information in the provided context.
- Be concise, precise, and factual.
- If the answer is a number, return only the number and unit.
- If the answer cannot be found in the context, say: "The information is not contained in the provided document."
"""

# User prompt = question + retrieved chunk
user_prompt = f"""
Question:
{prompt}

Context:
{doc.page_content}
"""

# Call the local LLM
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

