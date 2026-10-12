
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "features" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing = [key for key in required if not os.getenv(key)]
if missing:
    raise RuntimeError(f"Missing .env settings: {missing}")

url = URL.create(
    "mysql+pymysql",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    database=os.environ["DB_NAME"],
)

engine = create_engine(url)

query = """
SELECT f.*, l.target
FROM customer_features_v2 f
INNER JOIN customer_labels l
    ON l.customer_ID = f.customer_ID
ORDER BY f.customer_ID
"""

print("Reading customer-level features from MariaDB...")
df = pd.read_sql_query(query, engine)
engine.dispose()

if len(df) != 458_913:
    raise RuntimeError(f"Expected 458,913 rows, found {len(df):,}")

if df["customer_ID"].isna().any() or df["customer_ID"].duplicated().any():
    raise RuntimeError("Customer IDs are missing or duplicated.")

if df["target"].isna().any() or not set(df["target"].unique()).issubset({0, 1}):
    raise RuntimeError("Target must contain only 0 and 1, with no missing values.")

# The ID is a join key, never a model predictor.
customer_ids = df["customer_ID"].copy()
y = df["target"].astype("int8")

# Convert categorical features to pandas category dtype before splitting,
# so all partitions share the same category definitions.
categorical_features = [
    "latest__B_30",
    "latest__B_38",
    "latest__D_63",
    "latest__D_64",
    "latest__D_66",
    "latest__D_68",
    "latest__D_114",
    "latest__D_116",
    "latest__D_117",
    "latest__D_120",
    "latest__D_126",
]

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Keep target separate from predictors.
X = df.drop(columns=["customer_ID", "target"])
del df

# 70% train, 30% temporary holdout.
X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
    X, y, customer_ids,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Split the temporary holdout equally: 15% validation, 15% test.
X_valid, X_test, y_valid, y_test, id_valid, id_test = train_test_split(
    X_temp, y_temp, id_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

splits = {
    "train": (X_train, y_train, id_train),
    "validation": (X_valid, y_valid, id_valid),
    "test": (X_test, y_test, id_test),
}

all_ids = []
for name, (features, target, ids) in splits.items():
    out = features.copy()
    out.insert(0, "customer_ID", ids.to_numpy())
    out["target"] = target.to_numpy()

    path = OUTPUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False, compression="zstd")

    default_count = int(target.sum())
    print(
        f"{name:10s} rows={len(out):,} "
        f"defaults={default_count:,} "
        f"default_pct={target.mean() * 100:.2f}% "
        f"file={path}"
    )
    all_ids.extend(ids.tolist())

# Verify no customer occurs in more than one partition.
if len(all_ids) != len(set(all_ids)):
    raise RuntimeError("Customer leakage: IDs overlap between splits.")

if sum(len(item[1]) for item in splits.values()) != 458_913:
    raise RuntimeError("Split row counts do not reconcile.")

print("\nSplit validation passed.")
print("Predictor columns:", X_train.shape[1])
print("Customer IDs overlap across splits: 0")
