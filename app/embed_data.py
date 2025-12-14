from rag import build_embeddings, index

print("Starting embedding process...")

# Get current stats
try:
    stats = index.describe_index_stats()
    print(f"Current index stats: {stats}")

    namespaces = stats.get('namespaces', {})

    if namespaces:
        # Delete from each namespace
        for namespace in namespaces.keys():
            print(f"Deleting from namespace: '{namespace}'")
            index.delete(delete_all=True, namespace=namespace)
            print(f"✓ Deleted vectors from namespace: '{namespace}'")
    else:
        # Delete from default namespace
        print("Deleting from default namespace...")
        index.delete(delete_all=True, namespace="")
        print("✓ Deleted vectors from default namespace")

except Exception as e:
    print(f"Could not delete: {e}")
    print("Continuing anyway...")

# Embed 50 talks
build_embeddings(subset_size=50)

print("Embedding complete! Pinecone is ready.")