import os
import requests
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv
from app.utils import load_ted_data, chunk_text
import time
import re


# Load environment variables from .env file
load_dotenv()

# --- Load environment variables ---
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ted-rag-index")

# --- LLMod API URLs ---
LLMOD_API_URL = "https://api.llmod.ai/chat/completions"
LLMOD_EMBED_URL = "https://api.llmod.ai/v1/embeddings"

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ted-rag-index")


# --- LLMod chat function ---
def llmod_chat(system_prompt, user_prompt):
    response = requests.post(
        LLMOD_API_URL,
        headers={
            "Authorization": f"Bearer {LLMOD_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "RPRTHPB-gpt-5-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# --- LLMod embeddings function ---
def llmod_embed_batch(texts, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                LLMOD_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {LLMOD_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "RPRTHPB-text-embedding-3-small",
                    "input": texts,
                },
                timeout=60,
            )
            r.raise_for_status()
            return [x["embedding"] for x in r.json()["data"]]

        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)


# --- Build embeddings into Pinecone ---
def safe_meta(val):
    """Convert None/NaN to empty string for Pinecone metadata."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return val


def build_embeddings(subset_size=None, batch_size=16):
    df = load_ted_data(subset_size=subset_size)

    for _, row in df.iterrows():
        talk_id = str(row["talk_id"])
        transcript = row.get("transcript", "")
        if not transcript or pd.isna(transcript):
            continue

        chunks = chunk_text(transcript, chunk_size=1024, overlap_ratio=0.2)

        # Fetch all existing chunks for this talk at once
        vector_ids = [f"{talk_id}_{i}" for i in range(len(chunks))]
        existing = index.fetch(ids=vector_ids)["vectors"]
        existing_ids = set(existing.keys())

        texts_to_embed = []
        vectors_to_upsert = []

        for i, chunk in enumerate(chunks):
            vector_id = f"{talk_id}_{i}"

            # Skip if chunk already exists
            if vector_id in existing_ids:
                continue

            # Include ALL metadata in embedding text
            metadata_text = (
                f"Title: {safe_meta(row['title'])}. "
                f"Speaker: {safe_meta(row['speaker_1'])}. "
                f"All Speakers: {safe_meta(row['all_speakers'])}. "
                f"Occupations: {safe_meta(row['occupations'])}. "
                f"About Speakers: {safe_meta(row.get('about_speakers'))}. "
                f"Topics: {safe_meta(row['topics'])}. "
                f"Description: {safe_meta(row.get('description'))}. "
                f"Event: {safe_meta(row.get('event'))}. "
                f"Native Language: {safe_meta(row['native_lang'])}. "
                f"Available Languages: {safe_meta(row['available_lang'])}. "
                f"Recorded Date: {safe_meta(row.get('recorded_date'))}. "
                f"Published Date: {safe_meta(row.get('published_date'))}. "
                f"Views: {int(row['views']) if not pd.isna(row['views']) else 0}. "
                f"Duration: {int(row['duration']) if not pd.isna(row['duration']) else 0} seconds. "
            )

            text_to_embed = metadata_text + chunk
            texts_to_embed.append(text_to_embed)

            metadata = {
                "talk_id": talk_id,
                "title": safe_meta(row["title"]),
                "speaker_1": safe_meta(row["speaker_1"]),
                "all_speakers": safe_meta(row["all_speakers"]),
                "occupations": safe_meta(row["occupations"]),
                "about_speakers": safe_meta(row.get("about_speakers")),
                "views": int(row["views"]) if not pd.isna(row["views"]) else 0,
                "recorded_date": safe_meta(row.get("recorded_date")),
                "published_date": safe_meta(row.get("published_date")),
                "event": safe_meta(row.get("event")),
                "native_lang": safe_meta(row.get("native_lang")),
                "available_lang": safe_meta(row.get("available_lang")),
                "comments": safe_meta(row.get("comments")),
                "duration": int(row["duration"]) if not pd.isna(row["duration"]) else 0,
                "topics": safe_meta(row.get("topics")),
                "related_talks": safe_meta(row.get("related_talks")),
                "url": safe_meta(row.get("url")),
                "description": safe_meta(row.get("description")),
                "chunk_id": i,
                "chunk": chunk,
            }
            vectors_to_upsert.append((vector_id, metadata))

            # Batch upload
            if len(texts_to_embed) == batch_size:
                embeddings = llmod_embed_batch(texts_to_embed)
                index.upsert([
                    (vid, emb, meta)
                    for (vid, meta), emb in zip(vectors_to_upsert, embeddings)
                ])
                texts_to_embed.clear()
                vectors_to_upsert.clear()

        # Remaining embeddings
        if texts_to_embed:
            embeddings = llmod_embed_batch(texts_to_embed)
            index.upsert([
                (vid, emb, meta)
                for (vid, meta), emb in zip(vectors_to_upsert, embeddings)
            ])


# --- Retrieve top-k chunks from Pinecone ---
def retrieve(query, top_k=15):
    query_vec = llmod_embed_batch([query])[0]
    return index.query(vector=query_vec, top_k=top_k, include_metadata=True)


CATEGORY_INSTRUCTIONS = {
    "FACT": (
        "Return a single TED talk that matches the criteria. "
        "Provide only the required factual fields explicitly requested. "
        "Be concise and direct."
    ),
    "MULTI_LIST": (
        "Return multiple distinct TED talks as requested. "
        "Provide ONLY the list in the exact format requested (e.g., numbered list, bullet points). "
        "Do NOT include explanations, justifications, or additional commentary. "
        "Return only the talk titles or information explicitly requested. "
        "Ensure each result is a different talk."
    ),
    "SUMMARY": (
        "Identify one relevant TED talk and provide a concise summary of its key idea "
        "based only on the retrieved transcript content."
    ),
    "RECOMMENDATION": (
        "Recommend one TED talk and justify the recommendation "
        "using evidence from the retrieved context."
    ),
}


def categorize_question(q: str):
    """
    Categorize the user question into one of the four RAG categories.
    Returns: "FACT", "MULTI_LIST", "SUMMARY", "RECOMMENDATION"
    If no clear pattern is found, returns None.
    """
    q_lower = q.lower()

    # --- Category 2: Multi-Result Topic Listing ---
    if re.search(r"\b(which talks|which ted talk|list of exactly|return a list)\b", q_lower):
        return "MULTI_LIST"

    # --- Category 3: Key Idea Summary Extraction ---
    if any(kw in q_lower for kw in ["summary of", "summarize"]):
        return "SUMMARY"

    # --- Category 4: Recommendation ---
    if any(kw in q_lower for kw in ["would you recommend", "recommend a ted talk", "suggest a ted talk"]):
        return "RECOMMENDATION"

    # No clear match
    return None


def categorize_with_llm(question):
    """
    If categorize_question fails, use LLM to determine category.
    Returns one of the 4 category keys.
    """
    system_prompt = (
        "You are a categorization assistant. "
        "Classify the user question into one of these categories based on TED data tasks:\n"
        "1. FACT: Precise Fact Retrieval\n"
        "2. MULTI_LIST: Multi-Result Topic Listing (up to 3 results)\n"
        "3. SUMMARY: Key Idea Summary Extraction\n"
        "4. RECOMMENDATION: Recommendation with Evidence-Based Justification\n"
        "Respond only with the category key: FACT, MULTI_LIST, SUMMARY, or RECOMMENDATION. "
        "Do not explain your choice."
    )
    user_prompt = f"Question: {question}"
    category = llmod_chat(system_prompt, user_prompt).strip().upper()
    if category in CATEGORY_INSTRUCTIONS:
        return category
    return None


def category_instruction(category):
    if category is None:
        return ""
    return CATEGORY_INSTRUCTIONS[category]


# --- Answer question using retrieved chunks ---
def answer_question(question, top_k=15):
    category = categorize_question(question)
    if category is None:
        # fallback to LLM categorization if not matched safely
        category = categorize_with_llm(question)

    results = retrieve(question, top_k)

    all_chunks = []
    for m in results["matches"]:
        md = m["metadata"]
        all_chunks.append({
            "talk_id": md["talk_id"],
            "title": md["title"],
            "chunk": md["chunk"],
            "score": float(m["score"]),
            **md
        })

    raw_context = "\n\n---\n\n".join(
        f"Title: {c['title']}\nTranscript: {c['chunk']}"
        for c in all_chunks
    )

    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
        "If the answer cannot be determined from the provided context, respond: \"I don't know based on the provided TED data.\" "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.\n"
        + category_instruction(category)
    )

    user_prompt = f"Context:\n{raw_context}\n\nQuestion: {question}"

    response = llmod_chat(system_prompt, user_prompt)

    # Deduplicate context by talk
    seen = set()
    context_for_response = []
    for c in all_chunks:
        if c["talk_id"] not in seen:
            seen.add(c["talk_id"])
            context_for_response.append({
                "talk_id": c["talk_id"],
                "title": c["title"],
                "chunk": c["chunk"],
                "score": c["score"],
            })

    return response, context_for_response, system_prompt, user_prompt
