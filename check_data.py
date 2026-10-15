import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
FILE = BASE_DIR / "government_scheme_navigator_cleaned.csv"

df = pd.read_csv(FILE)

print("--- CLEANED DATASET VALIDATION ---")

# 1. Check dataset size
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Check new preprocessing columns
print("\n--- NEW COLUMNS ---")
print("Missing search_text:", df["search_text"].isna().sum())
print(
    "Empty search_text:",
    df["search_text"].fillna("").str.strip().eq("").sum()
)
print(
    "Missing age flag:",
    df["age_information_missing"].isna().sum()
)
print(
    "Missing status flag:",
    df["status_needs_verification"].isna().sum()
)

# 3. Check the age values
print("\n--- AGE VALIDATION ---")
print("Minimum age below zero:", (df["min_age"] < 0).sum())
print("Maximum age below zero:", (df["max_age"] < 0).sum())
print(
    "Minimum age greater than maximum age:",
    (
        df["min_age"].notna()
        & df["max_age"].notna()
        & (df["min_age"] > df["max_age"])
    ).sum()
)

# 4. Display a few records for manual inspection
print("\n--- SAMPLE SEARCH TEXT ---")
for _, row in df.head(3).iterrows():
    print("\nScheme:", row["scheme_name"])
    print("Search text:", row["search_text"][:300])

# 5. Display the verification flags
print("\n--- VERIFICATION SUMMARY ---")
print(df["status_needs_verification"].value_counts(dropna=False))
print(df["age_information_missing"].value_counts(dropna=False))

print("\nValidation completed.")
print("\n--- FINAL VALIDATION CHECKS ---")

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print(
    "Empty search text:",
    df["search_text"].fillna("").str.strip().eq("").sum()
)

print(
    "Minimum age below zero:",
    (df["min_age"] < 0).sum()
)

print(
    "Maximum age below zero:",
    (df["max_age"] < 0).sum()
)

print(
    "Minimum age greater than maximum age:",
    (
        df["min_age"].notna()
        & df["max_age"].notna()
        & (df["min_age"] > df["max_age"])
    ).sum()
)