import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Konfigurasi API
API_URL = "http://localhost:8000"

# Fungsi untuk memanggil API
def call_predict_api(url):
    """Call FastAPI predict-dspace endpoint"""
    try:
        response = requests.post(f"{API_URL}/predict-dspace", 
                                json={"url": url}, 
                                timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def call_model_info_api():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting model info: {str(e)}")
        return None

def get_sample_urls():
    """Get sample URLs from the API"""
    try:
        response = requests.get(f"{API_URL}/sample-urls", timeout=10)
        response.raise_for_status()
        return response.json().get("sample_urls", [])
    except:
        # Return default sample URLs if API call fails
        return [
            "https://dspace.mit.edu/handle/1721.1/7582",
            "https://dspace.mit.edu/handle/1721.1/134813",
        ]

# Fungsi untuk visualisasi keywords
def visualize_keywords(keywords, weights=None):
    if not keywords:
        return None
        
    # Generate weights if not provided
    if weights is None:
        weights = [1.0 - (i*0.08) for i in range(len(keywords))]
        weights = [max(w, 0.1) for w in weights]
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Keyword': keywords,
        'Weight': weights[:len(keywords)]
    })
    
    # Sort by weight descending
    df = df.sort_values('Weight', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        x='Weight',
        y='Keyword',
        orientation='h',
        title='Topic Keywords',
        color='Weight',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Relevance",
        yaxis_title="",
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

# Main App
def main():
    st.set_page_config(
        page_title="DSpace Topic Predictor",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1e3a8a;
        }
        .topic-header {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .stButton > button {
            width: 100%;
        }
        .article-box {
            background-color: #f8f9fa;
            border-left: 5px solid #4f8bf9;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .keyword-chip {
            background-color: #e9ecef;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 15px;
            display: inline-block;
        }
        .similar-article {
            border: 1px solid #dee2e6;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .similar-article:hover {
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .confidence-meter {
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-bar {
            height: 100%;
            background-color: #4f8bf9;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header
    st.title("📚 FindTopic: DSpace MIT Topic Predictor")
    st.markdown("Analyze academic papers from MIT DSpace and predict their topics")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        model_info = call_model_info_api()
        
        if model_info and model_info.get("status") == "success":
            st.info(f"**Topics found**: {model_info.get('topic_count')}")
            st.info(f"**Model size**: {model_info.get('model_size_mb'):.2f} MB")
            
            # Show sample topics
            st.subheader("Sample Topics")
            sample_topics = model_info.get("sample_topics", {})
            for topic_id, keywords in sample_topics.items():
                with st.expander(f"Topic {topic_id}"):
                    st.write(", ".join(keywords))
        else:
            st.warning("⚠️ Could not retrieve model information")
        
        st.markdown("---")
        st.markdown("Made with ❤️ by MLOps Team")
    
    # Main content as tabs
    tabs = st.tabs(["Topic Prediction", "About"])
    
    with tabs[0]:
        # URL Input
        st.subheader("Enter DSpace MIT URL")
        
        # Sample URLs
        sample_urls = get_sample_urls()
        selected_sample = st.selectbox(
            "Select a sample URL or enter your own below:",
            [""] + sample_urls,
            format_func=lambda x: "Select a sample URL..." if x == "" else x
        )
        
        # URL input field - use the selected sample or let user type their own
        url_input = st.text_input("Or enter a DSpace MIT URL:", 
                                value=selected_sample if selected_sample else "",
                                placeholder="https://dspace.mit.edu/handle/...")
        
        # Prediction button
        if st.button("Predict Topic", key="predict_button"):
            if not url_input or not url_input.startswith("https://dspace.mit.edu"):
                st.error("Please enter a valid DSpace MIT URL")
            else:
                with st.spinner("Fetching and analyzing article..."):
                    result = call_predict_api(url_input)
                    
                    if result:
                        # Display article info
                        st.markdown("### Article Information")
                        article = result.get("article", {})
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="article-box">
                                <h4>{article.get('title')}</h4>
                                <p><strong>Authors:</strong> {article.get('authors', 'N/A')}</p>
                                <p><strong>Year:</strong> {article.get('year', 'N/A')}</p>
                                <p><strong>Abstract:</strong> {article.get('abstract_preview', 'No abstract available')}...</p>
                                <p><a href="{article.get('url')}" target="_blank">View original article</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display prediction results
                        prediction = result.get("prediction", {})
                        st.markdown("### Prediction Result")
                        
                        # Topic header with styling based on confidence
                        topic_id = prediction.get("topic_id")
                        topic_name = prediction.get("topic_name", f"Topic {topic_id}")
                        confidence = prediction.get("confidence", 0)
                        
                        if topic_id == -1:
                            st.warning("⚠️ This article was classified as an **outlier** (doesn't fit well with any existing topic)")
                        else:
                            st.markdown(f"""
                            <div class="topic-header">
                                <h3>Topic: {topic_name}</h3>
                                <div>Topic ID: {topic_id}</div>
                                <div>Confidence: {confidence:.2f}</div>
                                <div class="confidence-meter">
                                    <div class="confidence-bar" style="width: {min(confidence * 100, 100)}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display keywords visualization
                            st.subheader("Topic Keywords")
                            keywords = prediction.get("keywords", [])
                            
                            if keywords:
                                # Visualize keywords using plotly
                                fig = visualize_keywords(keywords)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Fallback to simple display
                                    st.markdown(
                                        "".join([f"<span class='keyword-chip'>{kw}</span>" for kw in keywords]),
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("No keywords available for this topic")
                            
                            # Similar articles section
                            st.subheader("Similar Articles")
                            similar_articles = result.get("similar_articles", [])
                            
                            if similar_articles:
                                for i, article in enumerate(similar_articles):
                                    st.markdown(f"""
                                    <div class="similar-article">
                                        <h4>{article.get('title')}</h4>
                                        <p>{article.get('abstract', '')[:150]}...</p>
                                        <a href="{article.get('url')}" target="_blank">View article</a>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No similar articles found")
                    else:
                        st.error("Failed to get prediction. Make sure the API server is running.")

    with tabs[1]:
        st.header("About This App")
        st.markdown("""
        ## DSpace MIT Topic Predictor

        This application uses a trained BERTopic model to analyze academic papers from the MIT DSpace repository 
        and predict their topics based on their content.

        ### How it works:

        1. **Data Collection**: The system collects academic papers from the MIT DSpace repository.
        2. **Preprocessing**: Text is cleaned and prepared for analysis.
        3. **Topic Modeling**: A BERTopic model identifies key topics in the academic papers.
        4. **Prediction**: New papers are classified into the discovered topics.

        ### Features:
        - Analyze any paper from DSpace MIT by URL
        - View topic keywords and confidence scores
        - Find similar papers within the same topic
        - Explore topic distribution across the corpus

        ### Technology Stack:
        - **FastAPI**: Backend API service for predictions
        - **Streamlit**: Interactive web interface
        - **BERTopic**: State-of-the-art topic modeling
        - **Sentence Transformers**: BERT-based embeddings for text
        """)

if __name__ == "__main__":
    main()