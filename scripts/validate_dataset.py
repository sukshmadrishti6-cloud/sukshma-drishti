import os
import sys
import pandas as pd

def validate_dataset():
    filepath = "data/raw/pima_diabetes.csv"
    
    # 1. File exists
    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at {filepath}")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    
    # 2. Expected columns and no unexpected columns
    expected_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    
    if list(df.columns) != expected_columns:
        print(f"ERROR: Column schema mismatch.\nExpected: {expected_columns}\nActual: {list(df.columns)}")
        sys.exit(1)
        
    # 3. Expected row count
    if len(df) != 768:
        print(f"ERROR: Expected 768 rows, found {len(df)}")
        sys.exit(1)
        
    # 4. Target values and class distribution
    if "Outcome" not in df.columns:
        print("ERROR: Target column 'Outcome' not found")
        sys.exit(1)
        
    unique_targets = set(df["Outcome"].unique())
    if unique_targets != {0, 1}:
        print(f"ERROR: Target 'Outcome' contains invalid values: {unique_targets}")
        sys.exit(1)
        
    value_counts = df["Outcome"].value_counts().to_dict()
    if value_counts.get(0) != 500 or value_counts.get(1) != 268:
        print(f"ERROR: Expected distribution {0: 500, 1: 268}, found {value_counts}")
        sys.exit(1)
        
    # 5. Numeric types
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    if len(non_numeric_cols) > 0:
        print(f"ERROR: Non-numeric columns found: {list(non_numeric_cols)}")
        sys.exit(1)
        
    print("SUCCESS: Dataset validation passed.")
    sys.exit(0)

if __name__ == "__main__":
    validate_dataset()

