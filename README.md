```mermaid
flowchart LR
    A[Embedding model] --> B[CustomEmbeddingFunction]
    B --> C[ChromaDB collection]
    C --> D[User query]
    D --> E[Top-k retrieved docs]
    E --> F[Prompt with context]
    F --> G[Qwen2.5-0.5B answer]
 
    classDef model fill:#7F77DD,stroke:#3C3489,color:#fff
    classDef db fill:#1D9E75,stroke:#085041,color:#fff
    classDef output fill:#D85A30,stroke:#712B13,color:#fff
 
    class A,B model
    class C,D db
    class E,F,G output
```


# $$Embedding\_Model$$

Built with `PyTorch Embedding` class, trained on a custom tiny dataset.

## $$Anchor \rightarrow Positive$$

```python
("capital of France", "The capital of France is Paris.")
   ↑ anchor                ↑ positive
```

## $$Docs \rightarrow Anchors$$

```python
docs = [
  "The Eiffel Tower is located in Paris, France.",
  "Cairo is the capital of Egypt.",
  "Python is a popular programming language for AI.",
]
anchors = [
  "Eiffel Tower.",
  "Capital of Egypt",
  "AI programming language"
]
```

## $$Mission$$

Give a **fixed knowledge model** external data via:

1. Data it wasn't trained on
2. RAG → less hallucination
3. Accurate answers grounded in docs/papers

## $$ChromaDB\ Usage$$

Wrap the trained model in a `CustomEmbeddingFunction` so Chroma can call it directly:

```python
import torch as t
from chromadb import Documents, EmbeddingFunction, Embeddings

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def __call__(self, input: Documents) -> Embeddings:
        self.model.eval()
        embeddings = []
        with t.no_grad():
            for text in input:
                ids = [self.vocab[w] for w in text.split(" ") if w in self.vocab]
                if not ids:
                    ids = [0]
                ids_tensor = t.tensor([ids])
                vec = self.model(ids_tensor)
                embeddings.append(vec.squeeze(0).tolist())
        return embeddings
```

Connect to Chroma Cloud and create a collection using this embedding function:

```python
import chromadb
import dotenv, os

dotenv.load_dotenv(".env")
key = os.getenv("CHROMA_API")

client = chromadb.CloudClient(
    api_key=key,
    tenant='c6adecdc-f9bc-4b97-8060-bcea5554c00d',
    database='rag_project_1'
)

c = CustomEmbeddingFunction(model=model, vocab=vocab)

collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=c,
)
```

Add docs and query:

```python
collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))],
)

results = collection.query(
    query_texts=["Capital of Egypt"],
    n_results=2,
)
```




## $$Result$$

Tested with **Qwen2.5-0.5B** → promising results
