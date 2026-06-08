import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD DATA
games = pd.read_csv("games.csv")

# 2. CLEAN DATA
games["genre"] = games["genre"].fillna("")
games["tags"] = games["tags"].fillna("")

# Combine features (lowercase for better matching)
games["features"] = (games["genre"] + " " + games["tags"]).str.lower()

# 3. VECTORIZE TEXT DATA
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(games["features"])

# 4. SIMILARITY MATRIX
similarity_matrix = cosine_similarity(feature_matrix)

# 5. RECOMMENDATION FUNCTION
def recommend_game(game_title, top_n=5):
    """
    Returns top N similar games based on genre + tags similarity.
    """

    # Find game index safely
    matches = games[games["title"] == game_title]

    if matches.empty:
        return {"error": "Game not found"}

    game_index = matches.index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(similarity_matrix[game_index]))

    # Sort by similarity (descending)
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the game itself
    recommendations = similarity_scores[1:top_n + 1]

    # Format output nicely
    result = []
    for i, score in recommendations:
        result.append({
            "title": games.iloc[i]["title"],
            "score": round(score, 3)
        })

    return result

# 6. TEST RUN
if __name__ == "__main__":
    game_name = "Shadow Escape"
    recommendations = recommend_game(game_name)

    print(f"\nRecommendations for '{game_name}':\n")

    if isinstance(recommendations, dict) and "error" in recommendations:
        print(recommendations["error"])
    else:
        for r in recommendations:
            print(f"{r['title']} (score: {r['score']})")