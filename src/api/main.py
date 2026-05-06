from fastapi import FastAPI
from api.routers import rag, search

app = FastAPI(title="API sobre pragas de soja")
app.include_router(search.router)
app.include_router(rag.router)


@app.get("/")
def root():
    return {"status": "online"}
