import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LOAD DATA
games = pd.read_csv("games.csv")
ratings = pd.read_csv("ratings.csv")

# CLEAN DATA
games["genre"] = games["genre"].fillna("")
games["tags"] = games["tags"].fillna("")
games["features"] = (games["genre"] + " " + games["tags"]).str.lower()

# CONTENT MODEL
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(games["features"])
content_sim = cosine_similarity(tfidf_matrix)

# POPULARITY SCORE (from ratings)
popularity = ratings.groupby("game_id")["rating"].mean()

# HYBRID RECOMMEND FUNCTION
def hybrid_recommend(selected_genres, top_n=5):
    selected_genres = [g.lower() for g in selected_genres]

    # filter by genre
    filtered = games[games["genre"].str.lower().isin(selected_genres)]

    if filtered.empty:
        return []

    indices = filtered.index.tolist()

    # content similarity boost
    scores = content_sim[indices].mean(axis=0)

    results = []

    for i, score in enumerate(scores):
        game_id = games.iloc[i]["game_id"]

        pop_score = popularity.get(game_id, 3) / 5  # normalize

        final_score = (0.6 * score) + (0.4 * pop_score)

        results.append((i, final_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    output = []
    for i, score in results[:top_n]:
        output.append((games.iloc[i]["title"], round(score, 3)))

    return output

def recommend_similar_game(game_title, top_n=5):
    
    if game_title not in games["title"].values:
        return []

    idx = games[games["title"] == game_title].index[0]

    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, score in sim_scores[1:top_n+1]:
        results.append((games.iloc[i]["title"], round(score, 3)))

    return results