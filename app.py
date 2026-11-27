"""
app.py - Streamlit GUI for the content-based recommender.

Uses exactly the same model as the report and 02_evaluate.py (recsys.py):
TF-IDF on product text -> rating-weighted user profile -> cosine similarity
-> exclude already-seen items -> top-5.

Run (from the project folder, next to recsys.py, interactions.csv, items.csv):
    pip install -r requirements.txt
    streamlit run app.py
"""
import pandas as pd
import streamlit as st

import recsys

st.set_page_config(page_title="Beauty Product Recommender", page_icon="💄", layout="wide")


# ------------------------------ model loading --------------------------------
@st.cache_resource(show_spinner="Building TF-IDF model (first run only) ...")
def load_model():
    """Load data, make the same 80/20 per-user split, fit the model on TRAIN."""
    inter, items = recsys.load_data()
    train, test = recsys.split_by_user(inter)
    model = recsys.ContentRecommender(items, train)
    return model, train


@st.cache_data
def load_results():
    """Evaluation table produced by 02_evaluate.py (optional)."""
    try:
        return pd.read_csv("results.csv", index_col=0)
    except FileNotFoundError:
        return None


model, train = load_model()
valid_users = sorted(model.liked)          # users with >=1 positive train rating


def user_history(user_id, n=10):
    """Items the user rated highly in the training data (what the profile is built from)."""
    hist = train[(train.user_id == user_id) & (train.rating >= recsys.LIKE_THRESHOLD)]
    hist = hist.merge(model.items[["parent_asin", "title", "main_category", "store"]],
                      on="parent_asin", how="left")
    return (hist.sort_values("rating", ascending=False)
                [["title", "main_category", "store", "rating"]]
                .head(n).reset_index(drop=True))


def show_recommendations(df):
    """Show the top-5 list as cards, then as a table."""
    for _, row in df.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**#{row['rank']}  {row['title']}**")
            c1.caption(f"Category: {row['main_category'] or '-'}  ·  "
                       f"Brand/Store: {row['store'] or '-'}  ·  ASIN: {row['parent_asin']}")
            # Plain markdown instead of st.metric: st.metric truncates in narrow columns
            c2.markdown(f"<div style='text-align:center'><small>Similarity</small><br>"
                        f"<span style='font-size:1.5rem;font-weight:600'>"
                        f"{row['similarity']:.3f}</span></div>", unsafe_allow_html=True)
    with st.expander("Show as table"):
        st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------- sidebar ----------------------------------
with st.sidebar:
    st.header("About the system")
    st.write("Content-based filtering on **Amazon Reviews 2023 – Beauty and Personal Care**.")
    st.markdown(
        "- **Features:** title, features, description, categories, store\n"
        "- **Vectorization:** TF-IDF (1–2 grams)\n"
        "- **Similarity:** cosine\n"
        f"- **Liked item:** rating ≥ {recsys.LIKE_THRESHOLD:g}"
    )
    st.metric("Users", f"{train.user_id.nunique():,}")
    st.metric("Products", f"{len(model.items):,}")
    results = load_results()
    if results is not None:
        st.subheader("Evaluation (test set)")
        # Transposed (metrics as rows) so it fits in the narrow sidebar
        table = results.drop(columns="users_evaluated", errors="ignore").T
        table.columns = ["TF-IDF" if c.startswith("Content") else "Random" for c in table.columns]
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

# ----------------------------------- main ------------------------------------
st.title("💄 Beauty Product Recommender")

tab_user, tab_item = st.tabs(["Recommend for a user", "Similar products (item-to-item)"])

# ---- user-based (required by the project guide) ----
with tab_user:
    st.write("Enter a **user ID** to get 5 new products similar to the ones the user rated highly.")

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    def pick_random_user():
        # Runs as a button callback, i.e. BEFORE the text box is drawn,
        # which is the only time Streamlit allows changing its value.
        st.session_state.user_id = pd.Series(valid_users).sample(1).iloc[0]

    col_in, col_ex = st.columns([4, 1])
    col_in.text_input("User ID", key="user_id", placeholder="e.g. " + valid_users[0])
    col_ex.button("Random example", use_container_width=True, on_click=pick_random_user)

    if st.button("Get recommendations", type="primary"):
        user_id = st.session_state.user_id.strip()
        if not user_id:
            st.warning("Please enter a user ID.")
        else:
            try:
                recs = model.recommend(user_id, recsys.TOP_K)
            except KeyError:
                st.error(f"Unknown user ID: `{user_id}`. "
                         "This user is not in the dataset. Try the 'Random example' button.")
            except ValueError as exc:
                st.warning(str(exc))
            else:
                left, right = st.columns([3, 2])
                with left:
                    st.subheader(f"Top {recsys.TOP_K} recommendations")
                    show_recommendations(recs)
                with right:
                    st.subheader("Products this user liked")
                    st.caption("The user profile is built from these (training data only).")
                    st.dataframe(
                        user_history(user_id)[["rating", "title", "main_category"]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            "rating": st.column_config.NumberColumn("Rating", format="%.0f ⭐",
                                                                    width="small"),
                            "title": st.column_config.TextColumn("Title", width="large"),
                            "main_category": st.column_config.TextColumn("Category"),
                        },
                    )

# ---- item-to-item (extra) ----
with tab_item:
    st.write("Pick a product to see the 5 most similar products by content.")
    labels = (model.items.title.str.slice(0, 90) + "  [" + model.items.parent_asin + "]")
    choice = st.selectbox("Product", options=range(len(model.items)),
                          format_func=lambda i: labels.iloc[i], index=None,
                          placeholder="Type to search a product title ...")
    if st.button("Find similar products"):
        if choice is None:
            st.warning("Please select a product first.")
        else:
            show_recommendations(model.similar_items(model.items.parent_asin.iloc[choice],
                                                     recsys.TOP_K))
