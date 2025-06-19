import json
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import os
from tqdm import tqdm

# Download NLTK resources if needed
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Add domain-specific stopwords 
academic_stop_words = {
    'abstract', 'introduction', 'conclusion', 'references', 'doi', 'journal',
    'paper', 'article', 'research', 'study', 'studies', 'author', 'authors',
    'published', 'university', 'press', 'vol', 'volume', 'issue', 'et', 'al'
}
stop_words.update(academic_stop_words)

def preprocess(text):
    """Preprocess text with spaCy - lemmatization and stopword removal"""
    if not text or pd.isna(text) or not isinstance(text, str):
        return ""
        
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and 
           len(token.text) > 2 and  
           token.text.lower() not in stop_words
    ]
    return " ".join(tokens)

def preprocess_data(input_file="data_api/data_raw/data_scrape.json", 
                   output_file="data_api/data_clean/data_clean.json",
                   use_title=True, 
                   use_abstract=True,
                   title_weight=1.5):
    """
    Preprocess data combining title and abstract with optional weighting.
    
    Args:
        input_file: Path to the input JSON file with raw data
        output_file: Path to save the cleaned data
        use_title: Whether to include the title text
        use_abstract: Whether to include the abstract text
        title_weight: Weight factor for title terms (repeat title words this many times)
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    
    # Check if required columns exist
    if 'Judul' not in df.columns:
        raise ValueError("'Judul' column not found in the data")
    if use_abstract and 'Abstrak' not in df.columns:
        print("Warning: 'Abstrak' column not found. Setting use_abstract=False")
        use_abstract = False
    
    # Calculate total documents
    total_docs = len(df)
    print(f"Preprocessing {total_docs} documents...")
    
    # Preprocess title
    print("Preprocessing titles...")
    df["Cleaned_Title"] = df["Judul"].apply(preprocess)
    
    # Preprocess abstract 
    if use_abstract:
        print("Preprocessing abstracts...")
        df["Cleaned_Abstract"] = df["Abstrak"].apply(preprocess)
    
    # Combine title and abstract with weighting
    if use_title and use_abstract:
        print(f"Combining title (weight={title_weight}) and abstract...")
        
        # Function to combine with weighting
        def combine_text(row):
            # Repeat title words to give them more weight
            weighted_title = " ".join([row["Cleaned_Title"]] * int(title_weight))
            
            if title_weight % 1 > 0:
                title_words = row["Cleaned_Title"].split()
                partial_count = int(len(title_words) * (title_weight % 1))
                if partial_count > 0:
                    weighted_title += " " + " ".join(title_words[:partial_count])
            
            # Combine with abstract
            if row["Cleaned_Abstract"]:
                return weighted_title + " " + row["Cleaned_Abstract"]
            else:
                return weighted_title
                
        df["Cleaned_Combined"] = df.apply(combine_text, axis=1)
        
        cleaned_data = df[["Cleaned_Title", "Cleaned_Abstract", "Cleaned_Combined"]].to_dict(orient="records")
        
        content_column = "Cleaned_Combined"
    elif use_title:
        cleaned_data = df[["Cleaned_Title"]].to_dict(orient="records")
        content_column = "Cleaned_Title"
    elif use_abstract:
        cleaned_data = df[["Cleaned_Abstract"]].to_dict(orient="records")
        content_column = "Cleaned_Abstract"
    else:
        raise ValueError("At least one of use_title or use_abstract must be True")
    
    avg_length = df[content_column].apply(lambda x: len(x.split())).mean()
    max_length = df[content_column].apply(lambda x: len(x.split())).max()
    min_length = df[content_column].apply(lambda x: len(x.split())).min()
    empty_docs = (df[content_column].apply(lambda x: len(x.split())) == 0).sum()
    
    print(f"\nPreprocessing complete:")
    print(f"  - Content column: {content_column}")
    print(f"  - Average document length: {avg_length:.1f} words")
    print(f"  - Shortest document: {min_length} words")
    print(f"  - Longest document: {max_length} words")
    print(f"  - Empty documents: {empty_docs} ({empty_docs/total_docs*100:.1f}%)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
    
    print(f"Cleaned data saved to {output_file}")
    return cleaned_data