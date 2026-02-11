"""Convert .numbers file to CSV."""
import os
from numbers_parser import Document

data_dir = os.path.join(os.path.dirname(__file__), "data")
numbers_path = os.path.join(data_dir, "training_data_english_generated.numbers")
csv_path = os.path.join(data_dir, "training_data_english_generated.csv")

if not os.path.exists(numbers_path):
    print(f"Error: {numbers_path} not found")
    exit(1)

doc = Document(numbers_path)
all_rows = []
for sheet in doc.sheets:
    for table in sheet.tables:
        rows = table.rows(values_only=True)
        if rows:
            all_rows.extend(rows)

if all_rows:
    import pandas as pd
    df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    df.to_csv(csv_path, index=False)
    print(f"Converted to {csv_path}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
else:
    print("No data found in .numbers file")
