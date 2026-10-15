
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "government_scheme_navigator_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "scheme_data_audit.csv"

df = pd.read_csv(INPUT_FILE)

# Create a separate audit file; do not modify the source data.
audit_columns = [
    "scheme_name",
    "category",
    "target_beneficiary",
    "min_age",
    "max_age",
    "gender",
    "income_or_eligibility_condition",
    "occupation_or_profile",
    "residence",
    "operational_status",
    "application_window_status",
    "status_as_of",
    "status_verification_note",
    "official_source",
    "application_url",
]

audit = df[audit_columns].copy()

audit["age_review_required"] = df["age_information_missing"]
audit["status_review_required"] = df["status_needs_verification"]

audit["review_notes"] = ""

audit.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("--- SCHEME AUDIT FILE CREATED ---")
print("Schemes included:", len(audit))
print("Columns:", len(audit.columns))
print("Age reviews required:", int(audit["age_review_required"].sum()))
print("Status reviews required:", int(audit["status_review_required"].sum()))
print("Saved to:", OUTPUT_FILE.name)
print("Original and cleaned datasets were not modified.")
