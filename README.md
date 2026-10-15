# AI Government Service Navigator

An AI-powered web app that helps Indian citizens discover government schemes that may match their situation. Users can type or speak a question in **English, Hindi or Marathi**, get a grounded AI answer, listen to it, and run a preliminary eligibility check.

**Live demo:** https://ai-government-service-navigator-wqipwcdmssayumvgensnco.streamlit.app/

> This tool gives preliminary guidance only. It is not an official eligibility decision. Always confirm scheme details on the official website before applying.

## Features

- **Semantic search.** Finds relevant schemes by meaning, not just keywords, using Sentence Transformers and ChromaDB, with keyword re-ranking.
- **Grounded AI answers.** A Groq-hosted LLM writes answers using only the retrieved scheme records. Missing details are reported as "not mentioned in the available data" instead of being guessed.
- **Voice input.** Record a question and have it converted to text with Whisper (via Groq).
- **Voice output.** Listen to the answer in English, Hindi or Marathi using neural text-to-speech (edge-tts).
- **Language detection.** The answer language follows the language of the question.
- **Eligibility checker.** Enter age, state, income, gender and profile to see schemes sorted into *Likely fit*, *Check details* and *Probably not*, with the reasons shown.
- **Official links.** Official source and application links are shown wherever the dataset has them.

## Tech stack

| Area | Technology |
|------|-----------|
| Interface | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB (persistent) |
| LLM and speech-to-text | Groq API (`openai/gpt-oss-120b`, `whisper-large-v3`) |
| Text-to-speech | edge-tts |
| Data handling | pandas |

## How it works

```
Question (typed or spoken)
        |
Speech-to-text (Whisper) if spoken
        |
Language detection (English / Hindi / Marathi)
        |
Semantic search in ChromaDB + keyword re-ranking
        |
Top matching scheme records
        |
Groq LLM writes an answer from those records only
        |
Answer shown + optional voice output
```

The eligibility checker runs separately. It compares the user's details with the age, residence, gender and income fields in the dataset, and labels each scheme as a likely fit, needs checking, or a probable mismatch.

## Project structure

```
AI-Government-Service-Navigator/
├── ui.py                  Streamlit interface (run this file)
├── app.py                 Backend: search, RAG answer, eligibility logic
├── semantic_search.py     Builds the ChromaDB vector database
├── preprocess_data.py     Cleans the raw dataset
├── audit_data.py          Data quality checks
├── check_data.py          Data checks
├── government_scheme_navigator_cleaned.csv   Cleaned dataset (295 records)
├── chroma_db/             Persistent vector database
├── requirements.txt       Python dependencies
└── .streamlit/config.toml Theme and server settings
```

## Run it locally

**1. Clone the repository**
```
git clone https://github.com/gayatrigore-cmyk/AI-Government-Service-Navigator.git
cd AI-Government-Service-Navigator
```

**2. Create and activate a virtual environment**
```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a file named `.env` in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a key from the [Groq console](https://console.groq.com). Never commit this file. It is already listed in `.gitignore`.

**5. Build the vector database (only if `chroma_db/` is missing)**
```
python semantic_search.py
```

**6. Start the app**
```
streamlit run ui.py
```
Then open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub (without `.env`).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repository, branch `main`, main file `ui.py`, Python 3.12.
3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. The first build takes several minutes.

## Example questions

- English: *I am a student from Maharashtra looking for scholarship assistance.*
- Hindi: *मैं एक किसान हूँ और मुझे खेती के लिए सब्सिडी चाहिए।*
- Marathi: *मी महाराष्ट्रातील विद्यार्थी आहे, मला शिष्यवृत्ती हवी आहे.*

## Limitations

- The dataset has 295 records and may be incomplete or out of date. Scheme status, amounts, deadlines and documents are not independently verified.
- Many records do not list the benefit amount, required documents, deadline or application steps. The app reports these as missing rather than guessing.
- The eligibility checker is a first-pass screening. Income limits are read from free text on a best-effort basis, so results can be wrong. A "Likely fit" result does not guarantee approval.
- The embedding model is English-focused, so Hindi and Marathi searches can be less accurate than English ones.
- Voice features need an internet connection. Voice output quality for Marathi and Hindi depends on the speech service.
- The free hosting tier sleeps after inactivity, so the first load can take about a minute.

## Future improvements

- Add verified amounts, documents and deadlines for popular schemes.
- Use a multilingual embedding model for better Hindi and Marathi search.
- Add more states and a regular data-refresh process.
- Add user feedback to improve results.

## Disclaimer

This project is for educational purposes. It is not affiliated with any government body. Scheme information may be outdated, so verify everything on the official government website.
