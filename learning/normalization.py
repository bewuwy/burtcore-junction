import pandas as pd
import os

def find_model(url):
    os.system(f"wget {url} -P ./learning/processing required/")

def extract_hate_speech_and_tweets(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return
    
    df = pd.read_csv(input_file)

    # Extract the 'hate_speech' and 'tweet' columns
    extracted_data = df[['hate_speech', 'tweet']]

    # Save the extracted data to a new CSV file
    extracted_data.to_csv(output_file, index=False)

def extract_hate_speech_and_tweets_to_tsv(input_file, output_file):
    """
    Extract the 'hate_speech' and 'tweet' columns from a CSV file and save them as a TSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Extract the 'hate_speech' and 'tweet' columns
        extracted_data = df[['hate_speech', 'tweet']]

        # Save the extracted data to a new TSV file
        extracted_data.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Data successfully written to {output_file}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Example usage
if __name__ == "__main__":
    find_model

    input_file = "./learning/processing required/labeled_data.csv"
    output_file_csv = "./learning/processing required/hate_speech_tweets.csv"
    output_file_tsv = "./learning/processing required/hate_speech_tweets.tsv"
    extract_hate_speech_and_tweets(input_file, output_file_csv)
    extract_hate_speech_and_tweets_to_tsv(input_file, output_file_tsv)