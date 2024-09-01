from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  

# Load precomputed data
content_sim = np.load('content_sim.npy')
collab_sim = np.load('collab_sim.npy')
user_anime_matrix = pd.read_csv('simulated_user_data.csv', index_col=0)
df = pd.read_csv('anime_model_data.csv')

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

@app.route('/recommend/existing', methods=['POST'])
def recommend_for_existing_user():
    data = request.json
    user_watch_history = data.get('watch_history', {})
    num_recommendations = data.get('num_recommendations', 5)

    if not user_watch_history:
        return jsonify({'error': 'No watch history provided'}), 400

    user_vector = np.zeros(user_anime_matrix.shape[1])

    for anime in user_watch_history:
        if anime in user_anime_matrix.columns:
            anime_index = user_anime_matrix.columns.get_loc(anime)
            user_vector[anime_index] = user_watch_history[anime]

    user_similarities = cosine_similarity([user_vector], user_anime_matrix.fillna(0))[0]
    similar_user_indices = user_similarities.argsort()[::-1][1:]

    recommended_anime = []

    for similar_user_idx in similar_user_indices:
        similar_user_id = user_anime_matrix.index[similar_user_idx]
        similar_user_ratings = user_anime_matrix.loc[similar_user_id]

        unwatched_anime = similar_user_ratings[similar_user_ratings > 0].index.difference(
            pd.Index(user_watch_history.keys())
        )
        recommended_anime.extend(unwatched_anime)

        if len(recommended_anime) >= num_recommendations:
            break

    return jsonify({'recommendations': recommended_anime[:num_recommendations]})

@app.route('/recommend/new', methods=['POST'])
def recommend_for_new_user():
    data = request.json
    preferred_genres = data.get('preferred_genres', [])
    liked_anime_titles = data.get('liked_anime_titles', [])
    num_recommendations = data.get('num_recommendations', 5)

    if not preferred_genres and not liked_anime_titles:
        return jsonify({'error': 'No preferences provided'}), 400

    input_vector = ""

    if preferred_genres:
        input_vector += " ".join(preferred_genres) + " "

    if liked_anime_titles:
        for title in liked_anime_titles:
            anime_row = df[df['title_romaji'] == title]
            if not anime_row.empty:
                input_vector += anime_row['combined_features'].values[0] + " "

    input_vector = input_vector if input_vector else ""

    df['combined_features'] = df['combined_features'].fillna('')

    input_tfidf = tfidf_vectorizer.transform([input_vector])
    content_sim_scores = cosine_similarity(input_tfidf, tfidf_vectorizer.transform(df['combined_features'])).flatten()
    similar_anime_indices = content_sim_scores.argsort()[::-1]

    recommended_anime = []
    seen_titles = set(liked_anime_titles)  
    seen_franchises = set(title.split(':')[0] for title in liked_anime_titles)  # Correctly initialize seen franchises

    for index in similar_anime_indices:
        anime_title = df.iloc[index]['title_romaji']
        anime_format = df.iloc[index]['format']  # Assume 'format' column exists in your dataset
        anime_main_title = anime_title.split(':')[0]  # Extract main title to handle different adaptations

        # Exclude input titles, movies, OVAs, specials, or different adaptations of the same anime
        if (
            anime_title not in seen_titles and
            anime_format not in ['Movie', 'OVA', 'Special'] and  # Filter out undesired formats
            anime_main_title not in seen_franchises  # Filter out different adaptations of the same anime
        ):
            seen_titles.add(anime_title)
            seen_franchises.add(anime_main_title)
            recommended_anime.append(anime_title)

        if len(recommended_anime) >= num_recommendations:
            break

    return jsonify({'recommendations': recommended_anime})


if __name__ == '__main__':
    app.run(debug=True)
