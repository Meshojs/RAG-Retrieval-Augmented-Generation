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

## $$Result$$

Tested with **Qwen2.5-0.5B** → promising results
