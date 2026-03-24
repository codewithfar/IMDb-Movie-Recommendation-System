import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path fix (same logic as before)
base_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(base_path, "data", "clean_movies.csv")

df = pd.read_csv(data_path)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_matrix = vectorizer.fit_transform(df['story'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Save files
model_path = os.path.join(base_path, "model")

pickle.dump(df, open(os.path.join(model_path, "data.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(model_path, "vectorizer.pkl"), "wb"))
pickle.dump(cosine_sim, open(os.path.join(model_path, "similarity.pkl"), "wb"))

print("✅ MODEL READY")