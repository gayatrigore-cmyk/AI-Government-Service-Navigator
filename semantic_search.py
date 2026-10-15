
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# 1. Set project paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "government_scheme_navigator_cleaned.csv"
DB_DIR = BASE_DIR / "chroma_db"


# 2. Load the cleaned dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded: {len(df)} schemes")


# 3. Load the pretrained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")


# 4. Connect to persistent ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name="government_schemes"
)


# 5. Prepare scheme documents and metadata
documents = (
    df["search_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

ids = [
    f"scheme_{i}"
    for i in range(len(df))
]

metadatas = []

for _, row in df.iterrows():
    metadatas.append({
        "scheme_name": str(row["scheme_name"]),
        "category": str(row["category"]),
        "government_level": str(row["government_level"]),
        "benefit": (
            str(row["benefit"])
            if pd.notna(row["benefit"]) else ""
        ),
        "eligibility": (
            str(row["income_or_eligibility_condition"])
            if pd.notna(row["income_or_eligibility_condition"])
            else ""
        ),
        "official_source": (
            str(row["official_source"])
            if pd.notna(row["official_source"]) else ""
        ),
        "application_url": (
            str(row["application_url"])
            if pd.notna(row["application_url"]) else ""
        ),
        "status_needs_verification": True,
    })


# 6. Build the vector database only when needed
existing_count = collection.count()

if existing_count == len(documents):
    print("Existing vector database found. Reusing saved embeddings.")
else:
    print("Building vector database for the first time...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Vector database updated successfully.")

print(f"Vectors stored in ChromaDB: {collection.count()}")
print(f"Database location: {DB_DIR}")


# 7. Improved semantic search with keyword reranking
import re

def search_schemes(query, top_k=5):
    query = query.strip()

    if not query:
        print("Please enter the assistance you need.")
        return

    total_schemes = collection.count()

    if total_schemes == 0:
        print("The scheme database is empty.")
        return

    # Convert the user's question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Retrieve more candidates before reranking
    candidate_count = min(20, total_schemes)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    # Remove common words that add little search value
    stop_words = {
        "i", "need", "want", "a", "an", "the", "to", "for",
        "my", "me", "in", "on", "of", "and", "is", "from",
        "government", "scheme", "schemes", "please", "help"
    }

    query_words = set(
        re.findall(r"[a-zA-Z]+", query.lower())
    ) - stop_words

    ranked_results = []

    for i, metadata in enumerate(results["metadatas"][0]):
        document = results["documents"][0][i] or ""
        distance = results["distances"][0][i]

        # Lower vector distance means a closer semantic match.
        semantic_score = 1 / (1 + max(0, distance))

        # Check how many useful query words occur in the scheme details.
        searchable_text = " ".join([
            metadata.get("scheme_name", ""),
            metadata.get("category", ""),
            document,
            metadata.get("benefit", ""),
        ]).lower()

        matched_words = [
            word for word in query_words
            if word in searchable_text
        ]

        keyword_score = (
            len(matched_words) / len(query_words)
            if query_words else 0
        )

        # Combine semantic similarity and keyword relevance.
        final_score = (
            0.65 * semantic_score
            + 0.35 * keyword_score
        )

        ranked_results.append({
            "metadata": metadata,
            "document": document,
            "distance": distance,
            "score": final_score,
            "matched_words": matched_words,
        })

    ranked_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print(f"\nSearch query: {query}")
    print(f"Retrieved {min(top_k, len(ranked_results))} potentially relevant schemes.")
    print("Results are ranked using semantic similarity and keyword matches.")
    print("This ranking does not confirm eligibility or current availability.")

    for i, item in enumerate(ranked_results[:top_k]):
        metadata = item["metadata"]

        print("\n" + "-" * 60)
        print(f"{i + 1}. {metadata.get('scheme_name', 'Unnamed scheme')}")
        print(f"Category: {metadata.get('category', 'Not specified')}")
        print(f"Government level: {metadata.get('government_level', 'Not specified')}")
        print(f"Vector distance: {item['distance']:.4f}")
        print(f"Reranking score: {item['score']:.4f}")

        print("\nBenefit:")
        print(metadata.get("benefit") or "Not specified.")

        print("\nEligibility information:")
        print(metadata.get("eligibility") or "Not specified.")

        print("\nRetrieved scheme details:")
        print(item["document"][:350] or "No description available.")

        if item["matched_words"]:
            print("Matched terms:", ", ".join(item["matched_words"]))

        source = metadata.get("official_source", "")
        application_url = metadata.get("application_url", "")

        if source:
            print(f"\nListed official source: {source}")

        if application_url.startswith("https://"):
            print(f"Listed application link: {application_url}")

        print(
            "\nVerification: Current operational status and "
            "application availability have not been verified."
        )
        print(
            "Check current official guidelines before relying "
            "on eligibility information."
        )


# 8. Run search when this file is executed directly
if __name__ == "__main__":
    question = input(
        "\nDescribe the government assistance you need: "
    )
    search_schemes(question)
