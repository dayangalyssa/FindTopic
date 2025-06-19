import json
import os
import mlflow
import pandas as pd
import argparse
import platform
import psutil
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from sklearn.metrics import silhouette_score
from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
from umap import UMAP
from hdbscan import HDBSCAN

# Start Prometheus server
PROMETHEUS_URL = "http://localhost:8001"
start_http_server(8001)

# Prometheus metrics with 'model_training' label

TOPICS_FOUND = Gauge('topic_model_topics_found', 'Number of topics found in latest training', ['model_training'])
COHERENCE_SCORE = Gauge('topic_model_coherence_score', 'Latest coherence score of trained model', ['model_training'])
SILHOUETTE_SCORE = Gauge('topic_model_silhouette_score', 'Latest silhouette score of trained model', ['model_training'])
DOCUMENT_COUNT = Gauge('topic_model_document_count', 'Number of documents in training dataset', ['model_training'])
OUTLIER_COUNT = Gauge('topic_model_outlier_count', 'Number of documents classified as outliers', ['model_training'])
OUTLIER_PERCENTAGE = Gauge('topic_model_outlier_percentage', 'Percentage of documents classified as outliers', ['model_training'])
MEMORY_USAGE = Gauge('topic_model_memory_usage_mb', 'Memory usage during training in MB', ['model_training'])
TRAINING_SUCCESS = Counter('topic_model_training_success_total', 'Number of successful training executions', ['model_training'])
TRAINING_FAILURES = Counter('topic_model_training_failures_total', 'Number of failed training executions', ['model_training'])


def train_topic_model(input_file, output_file, model_path, min_topic_size, use_probabilities):
    label = {"model_training": PROMETHEUS_URL}

    # Increment training count
  
    
    mlflow.set_experiment("BERTopic_Experiment")

    with mlflow.start_run(run_name=f"bertopic_run_min{min_topic_size}"):
        try:
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            MEMORY_USAGE.labels(**label).set(start_memory)

            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                df = pd.DataFrame(raw_data)

                if "Cleaned_Combined" in df.columns:
                    docs = df["Cleaned_Combined"].tolist()
                    content_column = "Cleaned_Combined"
                elif "Cleaned_Title" in df.columns:
                    docs = df["Cleaned_Title"].tolist()
                    content_column = "Cleaned_Title"
                else:
                    raise ValueError("No suitable content column found")

                DOCUMENT_COUNT.labels(**label).set(len(docs))

                mlflow.log_param("Document_Count", len(docs))
                mlflow.log_param("Content_Column", content_column)
                mlflow.log_param("Embedding_Model", "all-MiniLM-L6-v2")
                mlflow.log_param("Min_Topic_Size", min_topic_size)

                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                topic_model = BERTopic(
                                        embedding_model=embedding_model,
                                        min_topic_size=min_topic_size
                                    )
                # 2. Latih model seperti biasa
                print("Fitting BERTopic model...")
                topics, probs = topic_model.fit_transform(docs)

                # 3. TAMBAHKAN INI: Hitung dan tampilkan jumlah outlier awal
                outlier_count = sum(1 for t in topics if t == -1)
                outlier_percentage = (outlier_count / len(topics)) * 100
                print(f"Initial outliers: {outlier_count} ({outlier_percentage:.2f}%)")

                # 4. TAMBAHKAN INI: Kurangi outlier setelah model dilatih
                if outlier_percentage > 20:  # Jika outlier lebih dari 20%
                    print("Reducing outliers after training...")
                    try:
                        # Metode 1: Coba gunakan reduce_outliers jika tersedia
                        new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs)
                        print(f"Reduced outliers using reduce_outliers method")
                    except AttributeError:
                        # Metode 2: Jika reduce_outliers tidak ada, coba update_topics
                        try:
                            topic_model.update_topics(docs, topics=topics)
                            print(f"Updated topics to reduce outliers")
                        except:
                            print("Could not reduce outliers - using original topics")
        
                    # Hitung outlier setelah reduksi
                    new_outlier_count = sum(1 for t in new_topics if t == -1)
                    new_outlier_percentage = (new_outlier_count / len(new_topics)) * 100
                    print(f"After reduction: {new_outlier_count} outliers ({new_outlier_percentage:.2f}%)")
                    
                    # Gunakan topik yang baru jika jumlah outlier berkurang
                    if new_outlier_count < outlier_count:
                        topics = new_topics
                        print("Using reduced outlier topics")

                num_topics = len(set(topics)) - (1 if -1 in topics else 0)
                print(f"Found {num_topics} topics")
                TOPICS_FOUND.labels(**label).set(num_topics)

                outlier_count = sum(1 for t in topics if t == -1)
                outlier_percentage = (outlier_count / len(topics)) * 100
                OUTLIER_COUNT.labels(**label).set(outlier_count)
                OUTLIER_PERCENTAGE.labels(**label).set(outlier_percentage)
                mlflow.log_metric("Outlier_Count", outlier_count)
                mlflow.log_metric("Outlier_Percentage", outlier_percentage)

                print("Calculating coherence score...")
                texts = [simple_preprocess(doc) for doc in docs]
                dictionary = Dictionary(texts)
                topic_words = topic_model.get_topics()
                topic_keywords = [[word for word, _ in topic_words[topic]] for topic in topic_words if topic != -1]

                if topic_keywords:
                    coherence_model = CoherenceModel(topics=topic_keywords, texts=texts, dictionary=dictionary, coherence='c_v')
                    coherence_score = coherence_model.get_coherence()
                    print(f"Coherence Score: {coherence_score:.4f}")
                    COHERENCE_SCORE.labels(**label).set(coherence_score)
                    mlflow.log_metric("Coherence_Score", coherence_score)

                print("Calculating silhouette score...")
                valid_docs = [doc for doc, topic in zip(docs, topics) if topic != -1]
                valid_topics = [topic for topic in topics if topic != -1]
                if len(valid_docs) > 1 and len(set(valid_topics)) > 1:
                    valid_embeddings = embedding_model.encode(valid_docs)
                    silhouette = silhouette_score(valid_embeddings, valid_topics)
                    print(f"Silhouette Score: {silhouette:.4f}")
                    SILHOUETTE_SCORE.labels(**label).set(silhouette)
                    mlflow.log_metric("Silhouette_Score", silhouette)

                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                MEMORY_USAGE.labels(**label).set(current_memory)
                mlflow.log_metric("memory_usage_mb", current_memory)

                print(f"Saving model to {model_path}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                topic_model.save(model_path, save_embedding_model=False)

                df['Predicted_Topic'] = topics
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_json(output_file, orient="records", indent=4, force_ascii=False)

                mlflow.log_artifact(output_file)
                mlflow.log_artifact(model_path)

                print("Creating visualizations...")
                viz_dir = "visualization"
                os.makedirs(viz_dir, exist_ok=True)
                topic_model.visualize_topics().write_html(f"{viz_dir}/topics_overview.html")
                topic_model.visualize_barchart().write_html(f"{viz_dir}/barchart.html")
                topic_model.visualize_heatmap(top_n_topics=10).write_html(f"{viz_dir}/heatmap.html")
                topic_model.visualize_hierarchy().write_html(f"{viz_dir}/hierarchy.html")

                mlflow.log_artifact(f"{viz_dir}/topics_overview.html")
                mlflow.log_artifact(f"{viz_dir}/barchart.html")
                mlflow.log_artifact(f"{viz_dir}/heatmap.html")
                mlflow.log_artifact(f"{viz_dir}/hierarchy.html")

                mlflow.log_param("Python Version", platform.python_version())
                mlflow.log_param("Pandas Version", pd.__version__)
                
                print("Generating topic name mapping...")
                topic_infos = topic_model.get_topic_info()
                topic_names = {}

                for _, row in topic_infos.iterrows():
                    topic_id = row["Topic"]
                    if topic_id == -1:
                        topic_names[topic_id] = "Outlier"
                    else:
                        topic_words = topic_model.get_topic(topic_id)
                        keywords = [word for word, _ in topic_words[:3]]
                        topic_names[topic_id] = ", ".join(keywords).title()

                # Simpan ke JSON
                topic_name_file = os.path.join(os.path.dirname(output_file), "topic_names.json")
                with open(topic_name_file, "w", encoding="utf-8") as f:
                    json.dump(topic_names, f, indent=4, ensure_ascii=False)
                mlflow.log_artifact(topic_name_file)

                # Tambahkan ke dataframe
                df["Topic_Name"] = df["Predicted_Topic"].map(topic_names)
                df.to_json(output_file, orient="records", indent=4, force_ascii=False)
               
                TRAINING_SUCCESS.labels(**label).inc()
                print("Training completed successfully!")

                return {
                    "topics": topics,
                    "num_topics": num_topics,
                    "coherence_score": coherence_score if 'coherence_score' in locals() else None,
                    "silhouette_score": silhouette if 'silhouette' in locals() else None
                }

        except Exception as e:
            TRAINING_FAILURES.labels(**label).inc()
            print(f"Error: {str(e)}")
            mlflow.log_param("Error", str(e))
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERTopic with MLflow + Prometheus")
    parser.add_argument("--input_file", type=str, default="../data_api/data_clean/data_cleanv2.json")
    parser.add_argument("--output_file", type=str, default="../data_api/output/data_with_topics.json")
    parser.add_argument("--model_path", type=str, default="models/bertopic_model.pkl")
    parser.add_argument("--min_topic_size", type=int, default=10)
    parser.add_argument("--use_probabilities", action="store_true")

    args = parser.parse_args()

    print("Starting topic model training...")
    result = train_topic_model(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        min_topic_size=args.min_topic_size,
        use_probabilities=args.use_probabilities
    )

    print(f"Training finished! Found {result['num_topics']} topics")


    print(f"Prometheus metrics available at: {PROMETHEUS_URL}/metrics")
    print("MLflow UI available at: http://localhost:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
