import os
import urllib.request
import zipfile
import pandas as pd

def download_and_extract_data(url: str, extract_to: str):
    """
    Downloads the SMS Spam Collection dataset and extracts it.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    zip_path = os.path.join(extract_to, "smsspamcollection.zip")
    
    # Download the dataset if it doesn't already exist
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
    # Extract the dataset
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def load_data(data_dir: str = "data"):
    """
    Loads the SMSSpamCollection data into a pandas DataFrame.
    Returns: pandas DataFrame
    """
    file_path = os.path.join(data_dir, "SMSSpamCollection")
    
    if not os.path.exists(file_path):
        # Automatically download if missing
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        download_and_extract_data(url, data_dir)
        
    print(f"Loading data from {file_path}...")
    # The dataset is tab-separated with two columns: label and message.
    df = pd.read_csv(file_path, sep='\t', names=["label", "message"])
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Successfully loaded dataset with {len(df)} records.")
    print("Sample details:")
    print(df.head())
    print("\nDataset label distribution:")
    print(df['label'].value_counts())
