import pandas as pd

df = pd.read_parquet("hf://datasets/ucberkeley-dlab/measuring-hate-speech/measuring-hate-speech.parquet")

# Group by comment_id: calculate mean for numeric columns, keep first for non-numeric
df = df.groupby('comment_id', as_index=False).agg({
    **{col: 'mean' for col in df.select_dtypes(include='number').columns if col != 'comment_id'},
    **{col: 'first' for col in df.select_dtypes(exclude='number').columns if col != 'comment_id'}
})

df.to_csv('dataset.csv', index=False)
