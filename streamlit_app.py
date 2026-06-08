import streamlit as st
from app import hybrid_recommend, recommend_similar_game, games

st.title("🎮 Advanced Game Recommendation System")

tab1, tab2 = st.tabs(["🎯 Genre-Based", "🔍 Search Game"])

# GENRE-BASED RECOMMENDATION
with tab1:
    st.subheader("Recommend by Genres")

    genres = sorted(games["genre"].dropna().unique().tolist())

    selected = st.multiselect("Select Genres", genres)

    top_n = st.slider("Top N", 1, 10, 5)

    if st.button("Recommend by Genre"):
        results = hybrid_recommend(selected, top_n)

        for title, score in results:
            st.write(f"🎮 {title} — {score}")

# SEARCH
with tab2:
    st.subheader("Find Similar Games")

    game_list = games["title"].tolist()

    selected_game = st.selectbox("Choose a Game", game_list)

    top_n2 = st.slider("Top N Similar", 1, 10, 5, key="s2")

    if st.button("Find Similar"):
        results = recommend_similar_game(selected_game, top_n2)

        for title, score in results:
            st.write(f"🎮 {title} — {score}")