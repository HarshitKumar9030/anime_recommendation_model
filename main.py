import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import ast

# Load the anime dataset
df = pd.read_csv('anime_dataset_extended_final.csv')

# Extract the 'romaji' titles from the 'title' column
df['title_romaji'] = df['title'].apply(lambda x: ast.literal_eval(x).get('romaji') if pd.notnull(x) else '')

# Combine genres, description, and other relevant features into a single string to represent the 'vibe' of the anime
df['combined_features'] = df.apply(
    lambda x: ' '.join(ast.literal_eval(x['genres'])) + ' ' + x['description']
    if pd.notnull(x['genres']) and pd.notnull(x['description']) else '',
    axis=1
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Calculate Cosine Similarity for Content-Based Filtering
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the content similarity matrix
np.save('content_sim.npy', content_sim)

# Save the TF-IDF vectorizer model
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Simulate user data
num_users = 100
num_anime = df.shape[0]

np.random.seed(42)
user_anime_matrix = pd.DataFrame(
    np.random.choice([0, 1, 2, 3, 4, 5], size=(num_users, num_anime), p=[0.8, 0.05, 0.05, 0.05, 0.025, 0.025]),
    columns=df['title_romaji'][:num_anime],
    index=[f'User{i+1}' for i in range(num_users)]
)

# Save the simulated user data to a CSV file
user_anime_matrix.to_csv('simulated_user_data.csv')

# Calculate collaborative filtering similarity
collab_sim = cosine_similarity(user_anime_matrix.fillna(0))

# Save the collaborative similarity matrix
np.save('collab_sim.npy', collab_sim)

# Save the anime data
df.to_csv('anime_model_data.csv', index=False)

print("Model training completed.")
print(f"Content-based similarity matrix shape: {content_sim.shape}")
print(f"Collaborative filtering similarity matrix shape: {collab_sim.shape}")
