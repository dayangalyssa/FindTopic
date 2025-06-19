import json
import os
import threading
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup
import re
from scrape import scrape_data
from explore_data import preprocess_data
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Import your training function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.topic_modeling_tracking import train_topic_model 

app = FastAPI(
    title="Academic Paper Topic Mining API",
    description="API untuk scraping, preprocessing, dan prediksi topik paper akademik",
    version="1.0.0"
)

try:
    from explore_data import preprocess_data
except ImportError:
    def preprocess_data(**kwargs):
        return [{"sample": "data"}]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "model", "models", "bertopic_model.pkl"))

class DSpaceURLRequest(BaseModel):
    url: str

# Global variable to track training status
training_status = {
    "is_training": False,
    "progress": "idle",
    "last_result": None,
    "error": None
}

def run_training_background(input_file, output_file, model_path, min_topic_size):
    """Background task untuk menjalankan training"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["progress"] = "starting"
        training_status["error"] = None
        
        print("Starting background training...")
        training_status["progress"] = "training_model"
        
        result = train_topic_model(
            input_file=input_file,
            output_file=output_file,
            model_path=model_path,
            min_topic_size=min_topic_size,
            use_probabilities=False
        )
        
        training_status["progress"] = "completed"
        training_status["last_result"] = result
        training_status["is_training"] = False
        
        print("Training completed successfully!")
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["progress"] = "error"
        training_status["error"] = str(e)
        print(f"Training failed: {e}")

@app.get("/")
def home():
    return {"message": "Server berjalan dengan baik"}

@app.get("/scrape")
def run_scraping():
    data = scrape_data()
    return {"message": "Scraping selesai", "jumlah_data": len(data)}

@app.get("/preprocess")
def run_preprocessing(use_title: bool = True, use_abstract: bool = True, title_weight: float = 1.5):
    """Endpoint to run preprocessing with options for title and abstract combination"""
    cleaned = preprocess_data(
        use_title=use_title,
        use_abstract=use_abstract,
        title_weight=title_weight
    )
    return {
        "message": "Preprocessing completed",
        "settings": {
            "use_title": use_title,
            "use_abstract": use_abstract,
            "title_weight": title_weight
        },
        "data_count": len(cleaned),
        "sample": cleaned[:3]
    }

@app.get("/train")
def start_training(
    background_tasks: BackgroundTasks,
    min_topic_size: int = 10,
    input_file: str = "../data_api/data_clean/data_cleanv2.json",
    output_file: str = "../data_api/output/data_with_topics.json",
    model_path: str = "../models/bertopic_model.pkl"
):
    """Endpoint untuk memulai training topic model"""
    global training_status
    
    if training_status["is_training"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Training sudah berjalan", "status": training_status["progress"]}
        )
    
    if not os.path.exists(input_file):
        return JSONResponse(
            status_code=400,
            content={"error": f"File input tidak ditemukan: {input_file}"}
        )
    
    background_tasks.add_task(
        run_training_background,
        input_file, output_file, model_path, min_topic_size
    )
    
    return {
        "message": "Training dimulai di background",
        "parameters": {
            "min_topic_size": min_topic_size,
            "input_file": input_file,
            "output_file": output_file,
            "model_path": model_path
        },
        "note": "Gunakan /train/status untuk melihat progress"
    }

@app.get("/train/status")
def get_training_status():
    """Endpoint untuk melihat status training"""
    return {
        "is_training": training_status["is_training"],
        "progress": training_status["progress"],
        "last_result": training_status["last_result"],
        "error": training_status["error"]
    }

@app.get("/train/results")
def get_training_results():
    """Endpoint untuk melihat hasil training terakhir"""
    if training_status["last_result"] is None:
        return {"message": "Belum ada hasil training"}
    
    return {
        "message": "Hasil training terakhir",
        "result": training_status["last_result"],
        "timestamp": time.time()
    }

@app.get("/results")
def show_preprocessed():
    cleaned = preprocess_data()
    return {"message": "Hasil preprocessing", "data": cleaned[:5]}

@app.get("/view")
def view_scraped_data():
    path = "data_api/data_raw/data_scrape.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"message": "Data hasil scraping", "jumlah_data": len(data), "data": data[:5]}
    else:
        return {"error": "File data_scraping.json tidak ditemukan"}

@app.get("/view/topics")
def view_topic_results():
    """Endpoint untuk melihat hasil topic modeling"""
    output_file = "data_api/output/data_with_topics.json"
    
    if not os.path.exists(output_file):
        return {"error": "Hasil topic modeling tidak ditemukan. Jalankan training terlebih dahulu."}
    
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topics = [item.get("Predicted_Topic", -1) for item in data]
    unique_topics = list(set(topics))
    topic_counts = {topic: topics.count(topic) for topic in unique_topics}
    
    return {
        "message": "Hasil topic modeling",
        "total_documents": len(data),
        "total_topics": len(unique_topics) - (1 if -1 in unique_topics else 0),
        "topic_distribution": topic_counts,
        "sample_data": data[:3]
    }

# Endpoint untuk melihat nama-nama topic
@app.get("/view/topic-names")
def view_topic_names():
    """Endpoint untuk melihat nama-nama topic"""
    topic_names_file = "data_api/output/topic_names.json"
    
    if not os.path.exists(topic_names_file):
        return {"error": "File topic names tidak ditemukan"}
    
    with open(topic_names_file, "r", encoding="utf-8") as f:
        topic_names = json.load(f)
    
    return {
        "message": "Nama-nama topic",
        "topic_names": topic_names
    }

####PREDICTING TOPICS FROM DSPACE MIT URL####
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scrape import scrape_data
except ImportError:
    def scrape_data():
        return [{"sample": "data"}]

try:
    from explore_data import preprocess_data
except ImportError:
    def preprocess_data(**kwargs):
        return [{"sample": "data"}]
    from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_field_after_h5(soup, field_name):
    """Extract field value that appears after specific h5 heading in DSpace MIT pages"""
    divs = soup.find_all("div", class_=re.compile(r"item-page-field-wrapper"))
    for div in divs:
        h5 = div.find("h5")
        if h5 and field_name.lower() in h5.text.strip().lower():
            for sibling in h5.next_siblings:
                if isinstance(sibling, str):
                    text = sibling.strip()
                    if text:
                        return text
                elif sibling.name:
                    text = sibling.get_text(strip=True)
                    if text:
                        return text
    return ""

def fetch_dspace_article(url):
    """Fetch and parse article from DSpace MIT repository"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title_element = soup.find('h2', class_='page-header')
        title = title_element.text.strip() if title_element else ""
        
        # Extract abstract
        abstract = extract_field_after_h5(soup, "Abstract")
        
        # Extract authors
        authors = extract_field_after_h5(soup, "Author")
        
        # Extract year
        date_issued = extract_field_after_h5(soup, "Date issued")
        year = ""
        if date_issued:
            match = re.search(r'\b(19|20)\d{2}\b', date_issued)
            if match:
                year = match.group(0)
        
        # Combine text for prediction
        combined_text = f"{title} {abstract}"
        
        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "url": url,
            "combined_text": combined_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching article: {str(e)}")

def load_topic_model():
    """Load BERTopic model from disk"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail=f"Model not found at {MODEL_PATH}")
        
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        model = BERTopic.load(MODEL_PATH, embedding_model=embedding_model)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def get_topic_names():
    """Load topic names from disk if available"""
    try:
        topic_names_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "data_api", "output", "topic_names.json")
        if os.path.exists(topic_names_path):
            with open(topic_names_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

# Endpoint untuk prediksi 
@app.post("/predict-dspace")
async def predict_from_dspace_url(request: DSpaceURLRequest):
    """
    Predict topic of a paper from DSpace MIT URL
    
    - Extracts text content from the provided URL
    - Performs topic prediction using trained BERTopic model
    - Returns predicted topic and related information
    """
    # Validate URL
    if not request.url.startswith("https://dspace.mit.edu"):
        raise HTTPException(status_code=400, detail="URL must be from DSpace MIT (https://dspace.mit.edu)")
    
    # Fetch article content
    article = fetch_dspace_article(request.url)
    
    if not article["combined_text"] or len(article["combined_text"]) < 10:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from the article")
    
    # Load model
    model = load_topic_model()
    
    # Get topic names
    topic_names = get_topic_names()
    
    # Predict topic
    try:
        topics, probs = model.transform([article["combined_text"]])
        topic_id = int(topics[0])
        
        if topic_id == -1:
            topic_name = "Outliers"
            keywords = []
            confidence = 0.0
        else:
            topic_info = model.get_topic(topic_id)
            keywords = [word for word, _ in topic_info[:10]]
            
            topic_name = topic_names.get(str(topic_id), f"Topic {topic_id}")
            
            confidence = 0.0
            try:
                if isinstance(probs, list) and len(probs) > 0:
                    if isinstance(probs[0], list):
                        topic_idx = list(model.get_topics().keys()).index(topic_id)
                        if topic_idx < len(probs[0]):
                            confidence = float(probs[0][topic_idx])
            except:
                pass
        
        # Get similar articles if available
        similar_articles = []
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "data_api", "output", "data_with_topics.json")
        if os.path.exists(data_path) and topic_id != -1:
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
                    
                similar_docs = [doc for doc in all_data if "Predicted_Topic" in doc and doc["Predicted_Topic"] == topic_id]
                
                for i, doc in enumerate(similar_docs[:3]):
                    similar_articles.append({
                        "title": doc.get("Judul", ""),
                        "abstract": doc.get("Abstrak", "")[:200] + "..." if len(doc.get("Abstrak", "")) > 200 else doc.get("Abstrak", ""),
                        "url": doc.get("URL", "")
                    })
            except:
                pass
        
        return {
            "article": {
                "title": article["title"],
                "abstract_preview": article["abstract"][:200] + "..." if len(article["abstract"]) > 200 else article["abstract"],
                "authors": article["authors"],
                "year": article["year"],
                "url": article["url"]
            },
            "prediction": {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "confidence": confidence,
                "keywords": keywords,
                "is_outlier": topic_id == -1
            },
            "similar_articles": similar_articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting topic: {str(e)}")

# Endpoint untuk info model
@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return {"status": "error", "message": f"Model not found at {MODEL_PATH}"}
        
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  
        model_modified = os.path.getmtime(MODEL_PATH)
        
        model = load_topic_model()
        topic_info = model.get_topic_info()
        topic_count = len(topic_info[topic_info["Topic"] != -1])
        
        sample_topics = {}
        count = 0
        for topic_id in sorted([t for t in model.get_topics().keys() if t != -1])[:5]:
            words = model.get_topic(topic_id)
            sample_topics[str(topic_id)] = [word for word, _ in words[:5]]
            count += 1
            if count >= 5:
                break
        
        return {
            "status": "success",
            "model_path": MODEL_PATH,
            "model_size_mb": round(model_size, 2),
            "topic_count": topic_count,
            "model_last_modified": model_modified,
            "sample_topics": sample_topics
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint untuk contoh URL DSpace MIT
@app.get("/sample-urls")
async def sample_dspace_urls():
    """Get sample DSpace MIT URLs for testing"""
    return {
        "sample_urls": [
            "https://dspace.mit.edu/handle/1721.1/39649",
            "https://dspace.mit.edu/handle/1721.1/71514",
            "https://dspace.mit.edu/handle/1721.1/62462"
        ]
    }

# Endpoint untuk simpel text prediction
@app.post("/predict-text")
async def predict_from_text(text: str):
    """Predict topic from raw text input"""
    if not text or len(text) < 20:
        raise HTTPException(status_code=400, detail="Text too short (minimum 20 characters required)")
    
    # Load model
    model = load_topic_model()
    topic_names = get_topic_names()
    
    # Predict
    try:
        topics, probs = model.transform([text])
        topic_id = int(topics[0])
        
        if topic_id == -1:
            return {
                "topic_id": -1,
                "topic_name": "Outliers",
                "keywords": [],
                "confidence": 0
            }
        else:
            topic_info = model.get_topic(topic_id)
            keywords = [word for word, _ in topic_info[:10]]
            topic_name = topic_names.get(str(topic_id), f"Topic {topic_id}")
            
            return {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "keywords": keywords,
                "confidence": float(probs[0].max()) if hasattr(probs[0], 'max') else 0.0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting topic: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)