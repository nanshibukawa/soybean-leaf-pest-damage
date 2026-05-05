import os
import uuid
import logging
from pathlib import Path
from qdrant_client import models
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

from rag.shared.constants import DENSE_MODEL, MAX_TOKENS, DATA_PATH, COLLECTION_NAME
from rag.shared.schemas import Chunk
from rag.ingestion.providers import EmbeddingProvider, VectorStoreProvider
from rag.ingestion.utils.chunker import SemanticChunker
from rag.ingestion.extractors.pypdf_extractor import PyPDFExtractor
from rag.ingestion.extractors.docling_extractor import DoclingExtractor

logger = logging.getLogger(__name__)


class Ingestor:
    """Orquestrador modular para extração, chunking e vetorização."""

    def __init__(self, extractor_type="docling", chunker_type="semantic"):
        self.extractor_type = extractor_type
        self.chunker_type = chunker_type

        # Infraestrutura
        self.vector_store = VectorStoreProvider()
        self.embedder = EmbeddingProvider()
        self.tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)
        self.model_max_length = self.tokenizer.model_max_length

        # Inicializa o extrator escolhido
        if extractor_type == "docling":
            self.extractor = DoclingExtractor()
        elif extractor_type == "pypdf":
            self.extractor = PyPDFExtractor()
        else:
            raise ValueError(f"Extractor desconhecido: {extractor_type}")

        # Inicializa o chunker uma única vez
        if chunker_type == "semantic":
            self.chunker_tool = SemanticChunker(max_tokens=MAX_TOKENS)
        else:
            self.chunker_tool = HybridChunker(
                tokenizer=DENSE_MODEL,
                max_tokens=MAX_TOKENS,
                merge_peers=True,
            )

    def _get_final_chunks(self, pdf_path: Path) -> list[Chunk]:
        """Executa a extração seguida do chunking."""
        data = self.extractor.extract(pdf_path)

        if self.chunker_type == "semantic":
            all_chunks = []
            chunks = self.chunker_tool.create_chunks(data.full_text)
            for chunk in chunks:
                all_chunks.append(
                    Chunk(
                        text=chunk,
                        page=0,
                        metadata={
                            "source": str(pdf_path.name),
                            "strategy": f"{self.extractor_type}_{self.chunker_type}",
                        },
                    )
                )
            return all_chunks
        else:
            if self.extractor_type != "docling":
                raise ValueError("O chunker 'structured' requer o extractor 'docling'.")

            doc_chunks = list(self.chunker_tool.chunk(data.doc_object))
            return [
                Chunk(
                    text=chunk.text,
                    page=(
                        chunk.meta.doc_items[0].prov[0].page_no
                        if chunk.meta.doc_items
                        else 0
                    ),
                    metadata={
                        "source": str(pdf_path.name),
                        "strategy": f"{self.extractor_type}_{self.chunker_type}",
                    },
                )
                for chunk in doc_chunks
            ]

    def process_file(self, pdf_path: Path):
        """Fluxo completo: Extração -> Chunking -> Embedding."""
        file_name = pdf_path.name
        chunks_data = self._get_final_chunks(pdf_path)

        points = []
        for item in chunks_data:
            text = item.text
            formatted_text = f"passage: {text}"

            # Truncagem de segurança
            tokens = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            if len(tokens) > self.model_max_length:
                formatted_text = self.tokenizer.decode(
                    tokens[: self.model_max_length - 2], skip_special_tokens=True
                )

            embeddings = self.embedder.generate_passage_embeddings(formatted_text)

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings,
                    payload={
                        "text": text,
                        "metadata": {
                            "source": file_name,
                            "page": item.page,
                            "strategy": f"{self.extractor_type}_{self.chunker_type}",
                        },
                    },
                )
            )
        return points

    def run(self):
        """Fluxo principal: Cria coleção e processa todos os PDFs da pasta de dados."""
        pdf_dir = Path(DATA_PATH)
        all_points = []

        print(f"🔍 Buscando PDFs em: {pdf_dir}")
        self.vector_store.create_collection()

        for pdf_file in pdf_dir.glob("*.pdf"):
            print(f"📄 Processando: {pdf_file.name}")
            try:
                points = self.process_file(pdf_file)
                all_points.extend(points)
            except Exception as e:
                print(f"❌ Erro ao processar {pdf_file.name}: {str(e)}")

        if all_points:
            print(
                f"⬆️ Fazendo upload de {len(all_points)} pontos para '{COLLECTION_NAME}'..."
            )
            self.vector_store.client.upload_points(
                collection_name=COLLECTION_NAME, points=all_points, batch_size=20
            )
            print(f"✅ Sucesso! {len(all_points)} chunks indexados.")
        else:
            print("⚠️ Nenhum arquivo processado ou nenhum chunk gerado.")


if __name__ == "__main__":
    pass
