import pandas as pd
import os

file_path = r"d:\SleepPause\Program\python\mosaic\切片\12.15\eval_text\gemini_batch_results_20251215_192241.xlsx"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print(f"Total rows: {len(df)}")
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
        # If there are specific columns like 'score' or 'success', print summary stats
        if 'score' in df.columns:
             print("\nScore Distribution:")
             print(df['score'].value_counts())
        
    except Exception as e:
        print(f"Error reading excel: {e}")
