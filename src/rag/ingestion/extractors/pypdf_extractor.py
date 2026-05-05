from pathlib import Path
from pypdf import PdfReader
from rag.shared.schemas import ExtractedData, PageData


class PyPDFExtractor:
    def extract(self, pdf_path: Path) -> ExtractedData:
        reader = PdfReader(pdf_path)
        pages = []
        full_text_list = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(PageData(text=text, page_no=i + 1))
            full_text_list.append(text)

        return ExtractedData(
            full_text="\n\n".join(full_text_list),
            pages=pages,
            doc_object=None,
            metadata={"source": str(pdf_path)},
        )


if __name__ == "__main__":
    extractor = PyPDFExtractor()
    pdf_path = "/home/nanshibukawa/Documents/mestrado/soybean-leaf-pest-damage/src/rag/data/documento375webIncluido.pdf"
    data = extractor.extract(pdf_path=pdf_path)
