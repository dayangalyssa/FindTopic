import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("../data/research_titles.csv")

def preprocess(text):
   
    doc = nlp(str(text))  
    
    print("\n===== SEBELUM =====")
    print(text)

    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    
    print("===== SESUDAH =====")
    print(tokens)  

    return " ".join(tokens)

df["Cleaned_Title"] = df["Title"].apply(preprocess)

df[["Title", "Cleaned_Title"]].to_csv("../data/dataset-cleaned.csv", index=False)

print("\nData setelah pre-processing disimpan di data/data_cleaned.csv")

df = pd.read_csv("../data/dataset-cleaned.csv")
print(df.head())  
