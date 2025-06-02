import pickle
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Load and preprocess data
df = pd.read_csv("tmdb_top_rated_by_genre_platforms.csv")
df['overview'] = df['overview'].fillna('')

# TF-IDF matrix creation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create index of titles
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation function
def get_recommendations(title, media_type='movie', top_n=5):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:30]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices]
    recommendations = recommendations[recommendations['media_type'] == media_type]

    return recommendations[['title', 'overview', 'poster_path', 'platforms']].head(top_n).to_dict(orient='records')


# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        title = request.form.get('title')
        media_type = request.form.get('media_type')
        recommendations = get_recommendations(title, media_type)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)