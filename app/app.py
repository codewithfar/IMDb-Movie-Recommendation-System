import streamlit as st
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Path fix
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model")

# Load files
df = pickle.load(open(os.path.join(model_path, "data.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(model_path, "vectorizer.pkl"), "rb"))
cosine_sim = pickle.load(open(os.path.join(model_path, "similarity.pkl"), "rb"))

# UI
st.title("🎬 Movie Recommendation System")

# Movie input
movie_name = st.text_input("Enter movie name:")

if st.button("Recommend by Movie"):

    df['movie'] = df['movie'].str.lower()
    movie_name = movie_name.lower()

    if movie_name in df['movie'].values:

        idx = df[df['movie'] == movie_name].index[0]

        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        st.subheader("Top 5 Similar Movies:")

        for i in scores[1:6]:
            st.write(df.iloc[i[0]]['movie'])

    else:
        st.error("Movie not found!")

# Story input
st.subheader("OR Search by Storyline")

user_input = st.text_area("Enter storyline:")

if st.button("Recommend by Story"):

    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, vectorizer.transform(df['story']))

    scores = list(enumerate(sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.subheader("Top 5 Movies:")

    for i in scores[:5]:
        st.write(df.iloc[i[0]]['movie'])