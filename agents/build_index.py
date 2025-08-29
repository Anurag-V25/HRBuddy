# agents/build_index.py
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Prefer the new package; fall back if needed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


def build_index(
    source_dir: str = "HR Documents",
    out_dir: str = "data/index",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    src = Path(source_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    print(f"🔎 Loading PDFs from: {src}")
    loader = DirectoryLoader(
        str(src),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    raw_docs = loader.load()
    print(f"📄 Loaded {len(raw_docs)} raw documents")

    if not raw_docs:
        print("⚠️ No PDFs found. Check the folder path or file extensions.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # IMPORTANT: normalize embeddings so FAISS L2 -> cosine via d2 = 2(1-cos)
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"🧠 Embeddings: {embed_model} (normalized)")

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(str(out))
    print(f"✅ FAISS index saved to: {out} (index.faiss, index.pkl)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build FAISS index from HR PDFs.")
    p.add_argument("--source", default="HR Documents")
    p.add_argument("--out", default="data/index")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--chunk_overlap", type=int, default=150)
    a = p.parse_args()
    build_index(a.source, a.out, a.model, a.chunk_size, a.chunk_overlap)