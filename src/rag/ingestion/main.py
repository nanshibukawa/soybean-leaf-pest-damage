import typer
from pathlib import Path
from dotenv import load_dotenv
from rag.ingestion.core import Ingestor

app = typer.Typer(help="Ingestão de documentos para o RAG de Agronomia.")

@app.command()
def ingest(
    extractor: str = typer.Option("docling", help="Tipo de extrator: 'docling' ou 'pypdf'"),
    chunker: str = typer.Option("semantic", help="Estratégia de chunking: 'semantic' ou 'structured'"),
):
    """
    Inicia o processo de ingestão de documentos.
    """
    load_dotenv()
    typer.echo(f"🚀 Iniciando ingestão (Extractor: {extractor}, Chunker: {chunker})...")
    
    ingestor = Ingestor(extractor_type=extractor, chunker_type=chunker)
    ingestor.run()


if __name__ == "__main__":
    app()
