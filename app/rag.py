import os
import requests
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv
from utils import load_ted_data, chunk_text

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
    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "RPRTHPB-gpt-5-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    response = requests.post(LLMOD_API_URL, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    # Extract the response from OpenAI-style format
    return result["choices"][0]["message"]["content"]


# --- LLMod embeddings function ---
def llmod_embed(text):
    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "RPRTHPB-text-embedding-3-small",
        "input": text
    }
    response = requests.post(LLMOD_EMBED_URL, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()

    # The API returns OpenAI-style format
    return result["data"][0]["embedding"]


# --- Build embeddings into Pinecone (subset only) ---
def build_embeddings(subset_size=50):
    """
    Load a subset of TED talks and create embeddings for Pinecone.
    """
    print(f"Starting embedding process for {subset_size} talks...")
    df = load_ted_data(subset_size=subset_size)

    for idx, row in df.iterrows():
        talk_id = str(row["talk_id"])
        transcript = row.get("transcript", "")
        if not transcript or pd.isna(transcript):
            continue

        chunks = chunk_text(transcript, chunk_size=1024, overlap_ratio=0.2)

        # Create metadata prefix to include in embedding
        metadata_text = (
            f"Title: {row['title']}. "
            f"Speaker: {row['speaker_1']}. "
            f"All Speakers: {row['all_speakers']}. "
            f"Occupations: {row['occupations']}. "
            f"About Speakers: {row.get('about_speakers', '')}. "
            f"Topics: {row['topics']}. "
            f"Description: {row.get('description', '')}. "
            f"Event: {row.get('event', '')}. "
            f"Native Language: {row['native_lang']}. "
            f"Available Languages: {row['available_lang']}. "
            f"Recorded Date: {row['recorded_date']}. "
            f"Published Date: {row['published_date']}. "
            f"Views: {row['views']}. "
            f"Duration: {row['duration']} seconds. "
        )

        for i, chunk in enumerate(chunks):
            metadata = {
                "talk_id": talk_id,
                "title": row["title"],
                "speaker_1": row["speaker_1"],
                "all_speakers": row["all_speakers"],
                "occupations": row["occupations"],
                "about_speakers": row["about_speakers"],
                "views": int(row["views"]) if not pd.isna(row["views"]) else 0,
                "recorded_date": row["recorded_date"],
                "published_date": row["published_date"],
                "event": row["event"],
                "native_lang": row["native_lang"],
                "available_lang": row["available_lang"],
                "comments": row["comments"],
                "duration": int(row["duration"]) if not pd.isna(row["duration"]) else None,
                "topics": row["topics"],
                "related_talks": row["related_talks"],
                "url": row["url"],
                "description": row["description"],
                "chunk_id": i,
                "chunk": chunk
            }
            text_to_embed = metadata_text + chunk
            vector = llmod_embed(text_to_embed)
            index.upsert([(f"{talk_id}_{i}", vector, metadata)])


# --- Retrieve top-k chunks from Pinecone ---
def retrieve(query, top_k=15):
    query_vector = llmod_embed(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results


# --- Answer question using retrieved chunks ---
def answer_question(question, top_k=15):
    retrieved = retrieve(question, top_k)

    # ALWAYS get all top-k chunks for the model to see (with ALL metadata)
    all_chunks = []
    for m in retrieved["matches"]:
        md = m["metadata"]
        all_chunks.append({
            "talk_id": md["talk_id"],
            "title": md["title"],
            "speaker_1": md.get("speaker_1", ""),
            "all_speakers": md.get("all_speakers", ""),
            "occupations": md.get("occupations", ""),
            "about_speakers": md.get("about_speakers", ""),
            "views": md.get("views", 0),
            "recorded_date": md.get("recorded_date", ""),
            "published_date": md.get("published_date", ""),
            "event": md.get("event", ""),
            "native_lang": md.get("native_lang", ""),
            "available_lang": md.get("available_lang", ""),
            "comments": md.get("comments", ""),
            "duration": md.get("duration", ""),
            "topics": md.get("topics", ""),
            "related_talks": md.get("related_talks", ""),
            "url": md.get("url", ""),
            "description": md.get("description", ""),
            "chunk": md["chunk"],
            "score": float(m["score"])
        })

    # Build context text from ALL chunks with ALL metadata (model sees everything)
    raw_context_text = "\n\n---\n\n".join([
        f"Title: {c['title']}\n"
        f"Speaker: {c['speaker_1']}\n"
        f"All Speakers: {c['all_speakers']}\n"
        f"Occupations: {c['occupations']}\n"
        f"About Speakers: {c['about_speakers']}\n"
        f"Views: {c['views']}\n"
        f"Recorded Date: {c['recorded_date']}\n"
        f"Published Date: {c['published_date']}\n"
        f"Event: {c['event']}\n"
        f"Native Language: {c['native_lang']}\n"  
        f"Available Languages: {c['available_lang']}\n"  
        f"Comments: {c['comments']}\n"  
        f"Related Talks: {c['related_talks']}\n"  
        f"Duration: {c['duration']}\n"
        f"Topics: {c['topics']}\n"
        f"URL: {c['url']}\n"
        f"Description: {c['description']}\n"
        f"Transcript: {c['chunk']}"
        for c in all_chunks
    ])

    # For the returned JSON context: deduplicate to unique talks
    # Return ONLY the required fields per assignment spec
    seen_talks = set()
    context_chunks_for_response = []

    for chunk in all_chunks:
        talk_id = chunk["talk_id"]
        if talk_id not in seen_talks:
            seen_talks.add(talk_id)
            # Only include required fields: talk_id, title, chunk, score
            context_chunks_for_response.append({
                "talk_id": talk_id,
                "title": chunk["title"],
                "chunk": chunk["chunk"],
                "score": chunk["score"]
            })

    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
        "If the answer cannot be determined from the provided context, respond: \"I don't know based on the provided TED data.\" "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."
    )
    user_prompt = f"Context:\n{raw_context_text}\n\nQuestion: {question}"
    response = llmod_chat(system_prompt, user_prompt)

    # Return deduplicated context in JSON response
    return response, context_chunks_for_response, system_prompt, user_prompt
