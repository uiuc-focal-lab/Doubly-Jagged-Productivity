import pandas as pd
from collections import Counter

# ---- config ----
path = "/Users/ggnds/Downloads/Combined Ratings (1).xlsx"
sheet_name = 0              # or e.g. "Sheet1"
usecols = "F,H,J,L,N,P"    # the 6 non-contiguous Excel columns
min_count = 3               # majority threshold
out_col = "majority_value"  # output column name
# ----------------

def majority_value(values, k=3):
    """Return a value that appears >= k times; else pd.NA. Ignores blanks/NaN."""
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        cleaned.append(v)
    if not cleaned:
        return pd.NA
    counts = Counter(cleaned)
    val, freq = counts.most_common(1)[0]
    return val if freq >= k else pd.NA

# Read the sheet (all columns), then select the 6 columns by letter
df = pd.read_excel(path, sheet_name="Combined",usecols=usecols)


df["majority_value"] = df.apply(lambda r: majority_value(r.tolist(), k=min_count), axis=1)

# write output (includes the six columns + majority)
df.to_excel("risks_with_majority.xlsx", index=False)
print("Wrote with_majority.xlsx")