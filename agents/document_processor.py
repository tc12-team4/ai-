"""
Agent 1 : Document Processor
Critère 1 (12%) : Extraction, Chunking et Embedding

Pipeline :
1. Extraction : PDFs → Markdown
2. Chunking : Sémantique par phrases (~500 tokens, overlap)
3. Embedding : SentenceTransformer (all-MiniLM-L6-v2)
4. Stockage : FAISS (cosine similarity) + métadonnées
"""

import re
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import random


class DocumentProcessor:
    """
    Processeur de documents Doxa KB
    Gère extraction, chunking sémantique, embedding et indexation vectorielle
    """

    def __init__(
        self,
        pdf_dir: str,
        md_dir: str = "./md_files",
        index_dir: str = "./vector_db",
        chunk_size: int = 500,
        overlap_sentences: int = 2
    ):
        self.pdf_dir = Path(pdf_dir)
        self.md_dir = Path(md_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

        self.md_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        print("Chargement du modèle d'embedding...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Modèle chargé")

        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.index = None

    # --------------------------------------------------
    # ÉTAPE 1 : Extraction PDF → Markdown
    # --------------------------------------------------
    def extract_pdfs_to_md(self) -> Dict[str, str]:
        md_contents = {}
        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print("Aucun PDF trouvé")
            return md_contents

        for pdf_file in pdf_files:
            try:
                reader = PdfReader(pdf_file)
                md_text = f"# {pdf_file.stem}\n\n"

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    text = re.sub(r'\s+', ' ', text).strip()
                    md_text += f"## Page {page_num}\n\n{text}\n\n"

                md_path = self.md_dir / f"{pdf_file.stem}.md"
                md_path.write_text(md_text, encoding="utf-8")

                md_contents[pdf_file.stem] = md_text
                print(f"{pdf_file.name} : {len(reader.pages)} pages extraites")

            except Exception as e:
                print(f"Erreur avec {pdf_file.name} : {e}")

        return md_contents

    # --------------------------------------------------
    # ÉTAPE 2 : Chunking sémantique par phrases
    # --------------------------------------------------
    def chunk_documents(self, md_contents: Dict[str, str]) -> Tuple[List[str], List[Dict]]:
        documents = []
        metadata = []

        for filename, text in md_contents.items():
            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "source": filename,
                    "chunk_id": i,
                    "length_tokens": len(chunk.split())
                })

            print(f"{filename} : {len(chunks)} chunks")

        self.documents = documents
        self.metadata = metadata
        return documents, metadata

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunking sémantique basé sur les phrases
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            tokens = len(sentence.split())

            if current_length + tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += tokens
            else:
                chunks.append(" ".join(current_chunk))

                # overlap : conserver les dernières phrases
                current_chunk = current_chunk[-self.overlap_sentences:]
                current_length = sum(len(s.split()) for s in current_chunk)

                current_chunk.append(sentence)
                current_length += tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # --------------------------------------------------
    # ÉTAPE 3 : Embeddings
    # --------------------------------------------------
    def create_embeddings(self) -> np.ndarray:
        if not self.documents:
            return np.array([])

        embeddings = self.embedder.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

        return embeddings

    # --------------------------------------------------
    # ÉTAPE 4 : Index FAISS (Cosine Similarity)
    # --------------------------------------------------
    def build_index(self, embeddings: np.ndarray):
        if embeddings.size == 0:
            return

        # Normalisation L2 → cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Index FAISS créé : {self.index.ntotal} vecteurs")

    # --------------------------------------------------
    # Sauvegarde / Chargement
    # --------------------------------------------------
    def save_index(self):
        if self.index is None:
            return

        faiss.write_index(self.index, str(self.index_dir / "doxa_kb.index"))

        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "chunk_size": self.chunk_size,
                "overlap_sentences": self.overlap_sentences
            }, f, ensure_ascii=False, indent=2)

        print("Index et métadonnées sauvegardés")

    def load_index(self) -> bool:
        index_path = self.index_dir / "doxa_kb.index"
        meta_path = self.index_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]

        print(f"Index chargé : {self.index.ntotal} vecteurs")
        return True

    # --------------------------------------------------
    # Pipeline complet
    # --------------------------------------------------
    def process(self):
        md_contents = self.extract_pdfs_to_md()
        if not md_contents:
            return

        self.chunk_documents(md_contents)
        embeddings = self.create_embeddings()
        self.build_index(embeddings)
        self.save_index()
        self.test_similarity()

    # --------------------------------------------------
    # Test de similarité
    # --------------------------------------------------
    def test_similarity(self, sample_size: int = 5):
        if self.index is None:
            return

        sample_indices = random.sample(
            range(len(self.documents)),
            min(sample_size, len(self.documents))
        )

        scores = []

        for idx in sample_indices:
            query = self.documents[idx]
            query_emb = self.embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_emb)

            score, _ = self.index.search(query_emb, k=1)
            similarity = score[0][0]
            scores.append(similarity)

            print(f"Chunk {idx} ({self.metadata[idx]['source']}): {similarity:.3f}")

        avg = sum(scores) / len(scores)
        print(f"Similarité moyenne : {avg:.3f}")


# --------------------------------------------------
# Fonction utilitaire
# --------------------------------------------------
def create_knowledge_base(pdf_dir: str, force_rebuild: bool = False):
    processor = DocumentProcessor(pdf_dir)

    if not force_rebuild and processor.load_index():
        return processor

    processor.process()
    return processor


if __name__ == "__main__":
    PDF_DIR = r"C:\Users\laptop spirit\Desktop\tc12"
    processor = create_knowledge_base(PDF_DIR, force_rebuild=True)
    print(f"KB prête : {processor.index.ntotal} vecteurs")
