# Dockerfile
FROM python:3.10

# Set workdir
WORKDIR /app

# Copy semua file
COPY . .

# Install dependency
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy & nltk resource
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en_core_web_sm

# Server
CMD ["uvicorn", "script.main:app", "--host", "0.0.0.0", "--port", "8000"]
