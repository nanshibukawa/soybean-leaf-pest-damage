import os
from groq import Groq
from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from rag.shared.constants import (
    DENSE_MODEL,
    SPARSE_MODEL,
    COLBERT_MODEL,
    MAX_TOKENS,
    DATA_PATH,
    COLLECTION_NAME,
)

load_dotenv()


qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

query_text = "O que é a Diabrostica speciosa ?? e como é feito o seu controle?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {
            "prefetch": [
                {"query": query_dense, "using": "dense", "limit": 10},
                {"query": query_sparse, "using": "sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)

max_score = max((result.score for result in results.points), default=1.0)

for r in results.points:
    normalized_score = r.score / max_score
    print(f"Score: {normalized_score}")
    print(f"Texto: {r.payload['text']}")
    print(f"Texto: {r.payload['metadata']}")

    print("-" * 80)

# --- Integração com Groq (RAG Final) ---

if results.points:
    # Inicializa o cliente com a chave de API
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Constrói o contexto incluindo a fonte e a página para maior rastreabilidade
    context_entries = []
    for i, r in enumerate(results.points):
        # Acessa os metadados corretamente de dentro do payload
        metadata = r.payload.get("metadata", {})
        source = metadata.get("source", "Desconhecida")
        page = metadata.get("page", "N/A")
        text = r.payload.get("text", "")
        context_entries.append(
            f"--- DOCUMENTO {i+1} (Fonte: {source}, Pág: {page}) ---\n{text}"
        )

    context = "\n\n".join(context_entries)

    # Chamada para o modelo Llama-3.3-70b
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em agronomia focado em pragas da soja e culturas associadas. "
                    "Sua tarefa é responder perguntas técnicas de forma precisa, utilizando exclusivamente o contexto fornecido. "
                    "Se a informação não estiver presente, responda que não possui dados suficientes nos documentos. "
                    "Sempre cite o nome do arquivo PDF da fonte e a página (ex: arquivo.pdf, Pág: X) ao afirmar algo."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{context}\n\nPERGUNTA: {query_text}",
            },
        ],
        temperature=0.1,  # Baixa temperatura para evitar alucinações
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    print("\n🤖 RESPOSTA DO ESPECIALISTA (GROQ):")
    print(completion.choices[0].message.content)

else:
    print(
        "\n⚠️ Nenhum resultado relevante foi encontrado na base de dados para responder a essa pergunta."
    )
