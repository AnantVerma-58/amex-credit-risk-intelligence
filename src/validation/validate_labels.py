
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data" / "raw" / "train_labels.csv"

labels = pd.read_csv(path)

assert labels["customer_ID"].notna().all()
assert labels["customer_ID"].is_unique
assert labels["target"].notna().all()
assert set(labels["target"].unique()) == {0, 1}

print("Label validation passed")
print(f"Customers: {len(labels):,}")
print("\nTarget distribution:")
print(labels["target"].value_counts().sort_index())
print("\nTarget proportions:")
print(labels["target"].value_counts(normalize=True).sort_index())