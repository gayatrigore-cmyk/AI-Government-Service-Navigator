import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "government_scheme_navigator_2026_rag_dataset.csv"
OUTPUT_FILE = BASE_DIR / "government_scheme_navigator_cleaned.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

# Standardize surrounding whitespace in text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

# Convert empty text values to missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Convert age columns to numeric values.
# Missing ages remain missing; they are not assumed to be zero.
for column in ["min_age", "max_age"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Identify records that need further verification
df["age_information_missing"] = (
    df["min_age"].isna() | df["max_age"].isna()
)

df["status_needs_verification"] = (
    df["operational_status"].fillna("").str.contains(
        "verify|confirmation", case=False, regex=True
    )
    |
    df["application_window_status"].fillna("").str.contains(
        "verify|not confirmed|varies", case=False, regex=True
    )
)

# Build searchable text for later keyword search and RAG preparation
search_columns = [
    "scheme_name",
    "scheme_description",
    "category",
    "target_beneficiary",
    "income_or_eligibility_condition",
    "occupation_or_profile",
    "residence",
    "benefit"
]

available_columns = [
    column for column in search_columns if column in df.columns
]

df["search_text"] = (
    df[available_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Do not automatically delete records or invent missing eligibility rules.
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("--- PREPROCESSING REPORT ---")
print("Original records:", original_rows)
print("Cleaned records:", len(df))
print("Columns after preprocessing:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Records with missing age information:",
      int(df["age_information_missing"].sum()))
print("Records needing status verification:",
      int(df["status_needs_verification"].sum()))
print("Missing values by column:")
print(df.isna().sum().to_string())
print("\nCleaned dataset saved to:", OUTPUT_FILE.name)
print("Original dataset was not modified.")