
# $$Embedding_Model$$
I used ```Pytorch Embedding class``` to create this model. I trained it on custom and tiny dataset.
<br>
### Anchors & Positives examples : 
```python 
("capital of France", "The capital of France is Paris.")
   ↑ anchor                ↑ positive
```
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
```go 
So the mission of that technique is to provide our (Fixed knowledge model) with data :
                                                       ↑     ↑        ↑

1 - that is not trained on 
2 - using RAG == Less hallucination 
3 - accurate answers based on (docs + papers etc)
```

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/79a070fc-f23f-48fa-862b-4fcd90b34df4" />
