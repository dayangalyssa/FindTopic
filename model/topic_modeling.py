import json
import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

def train_topic_model(input_file, output_file, model_path):
    # ================== 1. Load & siapkan data ==================
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)

    docs = df["Cleaned_Title"].tolist()

    # ================== 2. Training BERTopic ==================
    print("📦 Training BERTopic model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(docs)

    # ================== 3. Evaluasi: Coherence Score ==================
    print("🧪 Evaluasi coherence score...")
    
    texts = [simple_preprocess(doc) for doc in docs]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    topic_words = topic_model.get_topics()

    topic_keywords = [[word for word, _ in topic_words[topic]] for topic in topic_words if topic != -1]

    coherence_model = CoherenceModel(topics=topic_keywords, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"📈 Coherence Score: {coherence_score:.4f}")

    # ================== 4. Simpan model ==================
    print("💾 Menyimpan model ke folder: bertopic_model")
    os.makedirs("bertopic_model", exist_ok=True)
    topic_model.save(os.path.join("bertopic_model", "bertopic_model.pkl"), save_embedding_model=False)

    df['Predicted_Topic'] = topics
    df.to_json(output_file, orient="records", indent=4, force_ascii=False)

    print("✅ Selesai! Model tersimpan dan hasil telah diekspor.")

    # ================== 5. Visualisasi & Eksplorasi ==================
    print("🎨 Menyimpan visualisasi topik ke folder data/saved_model/")

    os.makedirs("../data/saved_model", exist_ok=True)

    topic_model.visualize_topics().write_html("../data/saved_model/topics_overview.html")
    topic_model.visualize_barchart().write_html("../data/saved_model/barchart.html")
    topic_model.visualize_heatmap(top_n_topics=10).write_html("../data/saved_model/heatmap.html")
    topic_model.visualize_hierarchy().write_html("../data/saved_model/hierarchy.html")

    print("✅ Visualisasi disimpan! Buka file HTML di folder 'saved_model/' untuk melihat hasilnya.")

if __name__ == "__main__":
    train_topic_model(
        input_file="../data/data_cleaned.json",
        output_file="../data/data_with_topics.json",
        model_path="bertopic_model/bertopic_model.pkl"
    )
