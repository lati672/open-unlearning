import os
import json
import pandas as pd

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

forget_ratio = '05'
# Path to your saves directory (adjust as needed)
base_dir = './saves'

base_dir = os.path.join(base_dir, f'tofu_forget{forget_ratio}')
# Collect all matching directories
methods = []
data = {}

# Loop through directories and load TOFU_SUMMARY.json
for method_dir in os.listdir(base_dir):
    method_path = os.path.join(base_dir, method_dir)
    eval_dir = os.path.join(method_path, 'evals')
    
    if os.path.isdir(method_path) and os.path.isdir(eval_dir):
        json_path = os.path.join(eval_dir, 'TOFU_SUMMARY.json')
        
        if os.path.exists(json_path):
            try:
                summary_data = load_json_data(json_path)
                methods.append(method_dir)
                
                for key, value in summary_data.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

# Convert data to a pandas DataFrame for better visualization
df = pd.DataFrame(data, index=methods)
df.to_csv('method_comparison.csv')
# Display the table
print(df)
