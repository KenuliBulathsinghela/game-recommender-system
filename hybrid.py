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

# CONTENT-BASED MODEL
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(games["features"])
content_similarity = cosine_similarity(feature_matrix)

# USER-ITEM MATRIX
user_item_matrix = ratings.pivot_table(
    index="user_id",
    columns="game_id",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# HYBRID RECOMMENDER
def hybrid_recommend(user_id, top_n=5, alpha=0.5):
    """
    alpha = weight for collaborative filtering
    (1-alpha) = weight for content-based
    """

    # -------- Collaborative Part --------
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)

    collab_scores = pd.Series(dtype=float)

    for other_user, sim in similar_users.items():
        user_ratings = user_item_matrix.loc[other_user]
        collab_scores = collab_scores.add(user_ratings * sim, fill_value=0)

    # -------- Content Part --------
    content_scores = pd.Series(0, index=games["game_id"])

    user_ratings = user_item_matrix.loc[user_id]

    liked_games = user_ratings[user_ratings >= 4].index

    for game_id in liked_games:
        idx = games[games["game_id"] == game_id].index[0]
        sim_scores = content_similarity[idx]

        for i, score in enumerate(sim_scores):
            content_scores[games.iloc[i]["game_id"]] += score

    # -------- Normalize --------
    if len(content_scores) > 0:
        content_scores = content_scores / content_scores.max()

    if len(collab_scores) > 0:
        collab_scores = collab_scores / collab_scores.max()

    # -------- Combine --------
    hybrid_scores = (alpha * collab_scores) + ((1 - alpha) * content_scores)

    # Remove already seen games
    seen = user_item_matrix.loc[user_id]
    hybrid_scores = hybrid_scores[seen == 0]

    # Top recommendations
    top_games = hybrid_scores.sort_values(ascending=False).head(top_n)

    results = []
    for game_id, score in top_games.items():
        title = games.loc[games["game_id"] == game_id, "title"].values[0]
        results.append((title, round(score, 3)))

    return results

def precision_at_k(user_id, k=5):
    recommended = hybrid_recommend(user_id, top_n=k)

    recommended_titles = [r[0] for r in recommended]

    # actual liked games (rating >= 4)
    actual = user_item_matrix.loc[user_id]
    actual_liked = actual[actual >= 4].index.tolist()

    actual_titles = games[games["game_id"].isin(actual_liked)]["title"].tolist()

    if len(actual_titles) == 0:
        return 0

    hits = len(set(recommended_titles) & set(actual_titles))

    return hits / k