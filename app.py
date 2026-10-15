"""Backend functions for AI Government Service Navigator.
No Streamlit UI is rendered from this module. Start the app with: streamlit run ui.py
"""
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from pathlib import Path
import os
import re
import pandas as pd
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "government_scheme_navigator_cleaned.csv"
DB_DIR = BASE_DIR / "chroma_db"
EMPTY_VALUES = {"", "nan", "none", "not specified", "n/a", "null"}
df = None
model = None
collection = None

def set_resources(dataframe, embedding_model, vector_collection):
    global df, model, collection
    df, model, collection = dataframe, embedding_model, vector_collection

def clean(value):
    text = str(value).strip()
    return "" if text.lower() in EMPTY_VALUES else text


def safe_url(value):
    value = clean(value)
    return value if value.startswith(("https://", "http://")) else ""




def load_data(file_path):
    data = pd.read_csv(file_path)
    if "search_text" in data.columns:
        data["search_text"] = data["search_text"].fillna("").astype(str).str.lower()
    return data


def load_search_resources(db_dir=None):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(db_dir or DB_DIR))
    return model, client.get_collection(name="government_schemes")


def detect_query_language(query, preferred_voice_language=None):
    """Detect English/Hindi/Marathi, including Hindi and Marathi written in Devanagari or Roman script."""
    if preferred_voice_language in {"English", "Hindi", "Marathi"}:
        # Voice transcription was explicitly requested in this language; preserve it.
        return preferred_voice_language
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Safe fallback for common scripts; Devanagari cannot reliably distinguish Hindi from Marathi.
        return "English" if not re.search(r"[\u0900-\u097F]", query) else "Hindi"
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Identify the primary language of the user's query. Reply with exactly one word: English, Hindi, or Marathi. If the query is mixed, choose the language used for the actual question. Hindi and Marathi may be written in Devanagari or Roman script."},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        detected = (response.choices[0].message.content or "").strip().lower()
        if "marathi" in detected:
            return "Marathi"
        if "hindi" in detected:
            return "Hindi"
        return "English"
    except Exception:
        return "English" if not re.search(r"[\u0900-\u097F]", query) else "Hindi"


def generate_rag_answer(user_query, retrieved_schemes, answer_language="English"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Groq API key not found. Please check your .env file."
    if not retrieved_schemes:
        no_results = {
            "English": "No relevant scheme records were found.",
            "Hindi": "कोई संबंधित योजना रिकॉर्ड नहीं मिला।",
            "Marathi": "संबंधित योजनांची नोंद आढळली नाही.",
        }
        return no_results.get(answer_language, no_results["English"])
    language_names = {"English": "English", "Hindi": "Hindi (हिन्दी)", "Marathi": "Marathi (मराठी)"}
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": (
                    "You are an AI Government Service Navigator for India. Use only the supplied scheme records. "
                    "Do not add outside facts or guess. If eligibility, benefits, documents, deadlines, or application "
                    "steps are absent, explain that this information is not mentioned in the available data. Never guarantee eligibility. "
                    "Name the scheme supporting each point and clearly mention missing/unverified information. "
                    f"IMPORTANT: Write the entire answer in {language_names.get(answer_language, 'English')}. "
                    "Do not answer in English when Hindi or Marathi is requested, except for official scheme names, URLs, and proper nouns. "
                    "Translate the final verification reminder into the same language. Keep the answer clear and suitable for being read aloud."
                )},
                {"role": "user", "content": f"User's situation:\n{user_query}\n\nRetrieved records:\n" + "\n\n".join(retrieved_schemes)},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or "The AI returned an empty answer."
    except Exception as exc:
        return f"The AI answer could not be generated. Check Groq configuration. ({type(exc).__name__}: {exc})"


STOP_WORDS = {
    "i", "am", "a", "an", "the", "is", "are", "was", "were", "to", "for", "from", "of", "and", "or",
    "in", "on", "at", "my", "me", "we", "our", "want", "need", "looking", "with", "please", "help",
    "government", "scheme", "schemes", "can", "could", "would", "have", "has", "had", "it",
}


def search_schemes(query, limit=5):
    embedded = model.encode(query, normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=[embedded], n_results=min(30, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    query_words = set(re.findall(r"\b[a-zA-Z]+\b", query.lower())) - STOP_WORDS
    ranked = []
    for i, result_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i] or {}
        doc = result["documents"][0][i] or ""
        distance = float(result["distances"][0][i])
        searchable = " ".join(str(meta.get(k, "") or "") for k in (
            "scheme_name", "category", "benefit", "eligibility", "target_beneficiary", "occupation_or_profile"
        )) + " " + doc
        doc_words = set(re.findall(r"\b[a-zA-Z]+\b", searchable.lower()))
        matched = query_words & doc_words
        keyword_score = len(matched) / len(query_words) if query_words else 0.0
        semantic_score = 1 / (1 + max(0, distance))
        ranked.append({"id": result_id, "metadata": meta, "document": doc, "distance": distance,
                       "score": 0.65 * semantic_score + 0.35 * keyword_score, "matched_words": sorted(matched)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


def get_row(item):
    try:
        idx = int(item["id"].split("_")[-1])
        return df.iloc[idx] if 0 <= idx < len(df) else None
    except (ValueError, IndexError, AttributeError):
        return None


def build_scheme_text(item, row):
    meta = item["metadata"]
    return (
        f"Scheme name: {clean(meta.get('scheme_name')) or clean(row.get('scheme_name')) or 'Not specified'}\n"
        f"Description: {clean(row.get('scheme_description')) or 'Not specified'}\n"
        f"Category: {clean(meta.get('category')) or clean(row.get('category')) or 'Not specified'}\n"
        f"Target beneficiaries: {clean(row.get('target_beneficiary')) or 'Not specified'}\n"
        f"Eligibility: {clean(meta.get('eligibility')) or clean(row.get('income_or_eligibility_condition')) or 'Not specified'}\n"
        f"Benefit: {clean(meta.get('benefit')) or clean(row.get('benefit')) or 'Not specified'}\n"
        f"Government level: {clean(meta.get('government_level')) or clean(row.get('government_level')) or 'Not specified'}\n"
        f"Official source: {clean(row.get('official_source')) or 'Not specified'}\n"
        f"Application link: {clean(row.get('application_url')) or 'Not specified'}\n"
        "The records may be incomplete or outdated. Do not infer missing facts."
    )


INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
]
STATUS_MATCH = "Potential match — verify official rules"
STATUS_VERIFY = "Needs manual verification"
STATUS_MISMATCH = "Potential mismatch"
PROFILE_KEYWORDS = {
    "Student": ["student", "scholarship", "education", "post-matric", "pre-matric", "fellowship", "aicte", "college", "university"],
    "Farmer": ["farmer", "agriculture", "cultivator", "kisan", "crop", "irrigation", "farm", "livestock"],
    "Job seeker": ["job seeker", "unemployed", "employment", "skill", "training", "vocational", "placement"],
    "Business owner": ["business", "entrepreneur", "enterprise", "mudra", "startup", "msme", "vendor", "self-employment"],
    "Person with disability": ["disability", "disabled", "divyang", "persons with disabilities"],
    "Other": [],
}


def parse_income_cap(text):
    text = clean(text).lower().replace(",", "").replace("₹", " ")
    if not any(term in text for term in ("income", "family income", "annual family")):
        return None
    if not re.search(r"not exceed|below|less than|up to|upto|within|maximum|under|not more than|limit|ceiling", text):
        return None
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?", text):
        value = float(number)
        if unit.startswith(("lakh", "lac")):
            value *= 100_000
        elif unit.startswith("crore"):
            value *= 10_000_000
        if value >= 10_000:
            return value
    return None


def parse_income_cap(text):
    text = clean(text).lower().replace(",", "").replace("₹", " ")
    if not any(term in text for term in ("income", "family income", "annual family")):
        return None
    if not re.search(r"not exceed|below|less than|up to|upto|within|maximum|under|not more than|limit|ceiling", text):
        return None
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?", text):
        value = float(number)
        if unit.startswith(("lakh", "lac")):
            value *= 100_000
        elif unit.startswith("crore"):
            value *= 10_000_000
        if value >= 10_000:
            return value
    return None


def assess_scheme(scheme, age, state, income, gender, profile):
    fails, unknowns, passes = [], [], []

    keywords = PROFILE_KEYWORDS.get(profile, [])
    profile_text = " ".join(clean(scheme.get(col, "")) for col in (
        "scheme_name", "target_beneficiary", "occupation_or_profile",
        "scheme_description", "category"
    )).lower()
    relevant = not keywords or any(word in profile_text for word in keywords)
    if keywords and relevant:
        passes.append(f"Matches your profile ({profile}).")

    # Age
    min_age = pd.to_numeric(scheme.get("min_age"), errors="coerce")
    max_age = pd.to_numeric(scheme.get("max_age"), errors="coerce")
    if pd.notna(min_age) and age < min_age:
        fails.append(f"Below the listed minimum age of {int(min_age)}.")
    elif pd.notna(max_age) and age > max_age:
        fails.append(f"Above the listed maximum age of {int(max_age)}.")
    elif pd.notna(min_age) or pd.notna(max_age):
        passes.append("Your age fits the listed range.")

    # State / residence
    residence = clean(scheme.get("residence", "")).lower()
    if not residence:
        unknowns.append("Residence rule not listed.")
    elif any(t in residence for t in ("india", "all states", "nationwide")):
        passes.append("Open to residents across India.")
    elif state.lower() in residence:
        passes.append(f"Available in {state}.")
    else:
        others = [s for s in INDIAN_STATES
                  if s.lower() in residence and s.lower() != state.lower()]
        if others:
            fails.append(f"Residence may be limited to: {', '.join(others)}.")
        else:
            unknowns.append("Residence rule needs a manual check.")

    # Gender
    sg = clean(scheme.get("gender", "")).lower()
    if gender in ("Female", "Male") and sg:
        f_target = bool(re.search(r"female|women|woman|girl", sg))
        m_target = bool(re.search(r"\bmale\b|\bmen\b|\bboy", sg))
        if gender == "Female" and m_target and not f_target:
            fails.append("The listed gender target may not match.")
        elif gender == "Male" and f_target and not m_target:
            fails.append("The listed gender target may not match.")
        elif f_target or m_target:
            passes.append("Gender matches the scheme's target.")

    # Income
    condition = clean(scheme.get("income_or_eligibility_condition", ""))
    income_unclear = False
    cap = parse_income_cap(condition) if condition else None
    if cap is not None:
        if income > cap:
            fails.append(f"Income may exceed the cap of about ₹{int(cap):,}.")
        else:
            passes.append(f"Your income is within about ₹{int(cap):,}.")
    elif "income" in condition.lower():
        income_unclear = True
        unknowns.append("An income rule exists but could not be read. Check it.")

    if fails:
        return STATUS_MISMATCH, fails, relevant
    if len(passes) >= 2 and not income_unclear:
        return STATUS_MATCH, passes + ["Verify the full rules on the official site."], relevant
    return STATUS_VERIFY, (unknowns + passes) or ["Not enough data to confirm."], relevant