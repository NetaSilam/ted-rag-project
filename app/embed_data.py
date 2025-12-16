from rag import build_embeddings

print("Starting embedding process for the full dataset...")

# Embed all TED talks
build_embeddings()  # No subset_size means full dataset

print("Embedding complete! Pinecone is now ready.")
