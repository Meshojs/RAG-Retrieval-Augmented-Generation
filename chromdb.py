import chromadb
import dotenv
import os
from main import outro, CustomEmbeddingFunction

vocab = outro.vocab
model = outro.model

dotenv.load_dotenv(".env")
key = os.getenv("CHROMA_API")

client = chromadb.CloudClient(
  api_key=key,
  tenant='...',
  database='rag_project_1'
)

c = CustomEmbeddingFunction(model=model, vocab=vocab)

collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=c,
)

# try:
#     collection.add(
#         documents=[
#             "The Eiffel Tower is located in Paris, France.",
#             "Cairo is the capital of Egypt.",
#             "Python is a popular programming language for AI.",
#         ],
#         ids=["doc_0", "doc_1", "doc_2"],
#     )
#     print("Added successfully")
# except Exception as e:
#     print("ERROR:", repr(e))

# print("Count:", collection.count())




results = collection.query(
    query_texts=["Capital of Egypt"],
    n_results=2,
)
print(results)
