import json
import os
import mlflow
import pandas as pd
import argparse
import platform
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from sklearn.metrics import silhouette_score


def train_topic_model(input_file, output_file, model_path, min_topic_size, use_probabilities):
    # ================== MLflow Setup ==================
    mlflow.set_experiment("BERTopic Experiment")
    with mlflow.start_run(run_name=f"bertopic_run_min{min_topic_size}"):

        try:
            # ========== 1. Load Data ==========
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            df = pd.DataFrame(raw_data)
            docs = df["Cleaned_Title"].tolist()

            mlflow.log_param("Document Count", len(docs))
            mlflow.log_param("Embedding Model", "all-MiniLM-L6-v2")

            # ========== 2. Train BERTopic ==========
            print("📦 Training BERTopic model...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=min_topic_size
            )
            mlflow.log_param("Min Topic Size", min_topic_size)

            try:
                topics, probs = topic_model.fit_transform(docs, calculate_probabilities=use_probabilities)
                mlflow.log_param("Use Probabilities", use_probabilities)
            except TypeError:
                print("⚠️ calculate_probabilities tidak dikenali. Gunakan versi terbaru BERTopic.")
                topics, probs = topic_model.fit_transform(docs)
                mlflow.log_param("Use Probabilities", False)

            # ========== 3. Coherence Score ==========
            print("🧪 Evaluasi coherence score...")
            texts = [simple_preprocess(doc) for doc in docs]
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            topic_words = topic_model.get_topics()
            topic_keywords = [[word for word, _ in topic_words[topic]] for topic in topic_words if topic != -1]

            coherence_model = CoherenceModel(topics=topic_keywords, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            print(f"📈 Coherence Score: {coherence_score:.4f}")
            mlflow.log_metric("Coherence Score", coherence_score)

            # ========== 4. Silhouette Score ==========
            print("📊 Menghitung silhouette score...")
            valid_docs = [doc for doc, topic in zip(docs, topics) if topic != -1]
            valid_topics = [topic for topic in topics if topic != -1]
            valid_embeddings = embedding_model.encode(valid_docs)

            silhouette = silhouette_score(valid_embeddings, valid_topics)
            mlflow.log_metric("Silhouette Score", silhouette)

            # ========== 5. Save Model ==========
            print("💾 Menyimpan model ke folder:", model_path)
            os.makedirs("bertopic_model", exist_ok=True)
            topic_model.save(model_path, save_embedding_model=False)

            df['Predicted_Topic'] = topics
            df.to_json(output_file, orient="records", indent=4, force_ascii=False)

            mlflow.log_artifact(output_file)
            mlflow.log_artifact(model_path)

            # ========== 6. Visualisasi ==========
            print("🎨 Menyimpan visualisasi topik ke folder data/saved_model/")
            os.makedirs("../data/saved_model", exist_ok=True)

            topic_model.visualize_topics().write_html("../data/saved_model/topics_overview.html")
            topic_model.visualize_barchart().write_html("../data/saved_model/barchart.html")
            topic_model.visualize_heatmap(top_n_topics=10).write_html("../data/saved_model/heatmap.html")
            topic_model.visualize_hierarchy().write_html("../data/saved_model/hierarchy.html")

            mlflow.log_artifact("../data/saved_model/topics_overview.html")
            mlflow.log_artifact("../data/saved_model/barchart.html")
            mlflow.log_artifact("../data/saved_model/heatmap.html")
            mlflow.log_artifact("../data/saved_model/hierarchy.html")

            # ========== 7. Env Info ==========
            mlflow.log_param("Python Version", platform.python_version())
            mlflow.log_param("Pandas Version", pd.__version__)

            print("✅ Proses selesai. Semua hasil telah dicatat ke MLflow.")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            mlflow.log_param("Error", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERTopic with MLflow Tracking")
    parser.add_argument("--input_file", type=str, default="../data/data_cleaned.json")
    parser.add_argument("--output_file", type=str, default="../data/data_with_topics.json")
    parser.add_argument("--model_path", type=str, default="bertopic_model/bertopic_model.pkl")
    parser.add_argument("--min_topic_size", type=int, default=10)
    parser.add_argument("--use_probabilities", action="store_true", help="Use calculate_probabilities in BERTopic")

    args = parser.parse_args()

    train_topic_model(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        min_topic_size=args.min_topic_size,
        use_probabilities=args.use_probabilities
    )
