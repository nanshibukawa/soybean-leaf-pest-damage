from pathlib import Path
from docling.document_converter import DocumentConverter
from rag.shared.schemas import ExtractedData


class DoclingExtractor:
    def __init__(self):
        self.converter = DocumentConverter()

    def extract(self, pdf_path: Path) -> ExtractedData:
        result = self.converter.convert(pdf_path)
        doc = result.document

        return ExtractedData(
            full_text=doc.export_to_markdown(),
            pages=[],
            doc_object=doc,
            metadata={"source": str(pdf_path)},
        )


if __name__ == "__main__":
    extractor = DoclingExtractor()
    pdf_path = "/home/nanshibukawa/Documents/mestrado/soybean-leaf-pest-damage/src/rag/data/documento375webIncluido.pdf"
    data = extractor.extract(pdf_path=pdf_path)
    data.full_text
