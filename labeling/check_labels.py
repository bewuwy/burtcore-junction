import numpy as np
import pandas as pd

original_file = 'dataset.csv'
filled_file = 'dataset_extremism_filled.csv'
col = 'extremism_filled'
pred_col = 'extremism_pred'

# Load both files
df_orig = pd.read_csv(original_file)
df_filled = pd.read_csv(filled_file)

# Find indices where the original column is null
null_indices = np.where(df_orig["extremism"].isnull())

# Compare predictions only at those indices
matches = (df_filled.loc[null_indices, col] == df_filled.loc[null_indices, pred_col]).sum()
total = len(null_indices)
percentage = (matches / total) * 100 if total > 0 else 0

print(f'Match percentage (on originally null rows): {percentage:.2f}% ({matches}/{total})')
