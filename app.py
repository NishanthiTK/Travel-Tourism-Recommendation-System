
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Travel & Tourism Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

DATA_PATH = BASE_DIR / "dataset" / "Travel_Destinations_Cleaned.csv"



# ==========================================
# 3. PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #10243A;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(120deg, #10243A, #087E8B);
    padding: 35px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #E0F2F1;
    font-size: 16px;
}

.metric-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    min-height: 110px;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #10243A;
    font-size: 28px;
    font-weight: bold;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD DATA AND MODELS
# ==========================================

@st.cache_resource
def load_models():

    recommendation_model = joblib.load(
        MODEL_DIR / "tourism_recommendation_model.pkl"
    )

    prediction_models = joblib.load(
        MODEL_DIR / "tourism_prediction_models.pkl"
    )

    return recommendation_model, prediction_models


@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    recommendation_model, prediction_models = load_models()

    clean_data = load_dataset()

except Exception as e:

    st.error(f"Unable to load project files: {e}")
    st.stop()


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("## ✈️ Travel Intelligence")

    st.caption("Machine Learning Based Tourism Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home Dashboard",
            "Destination Recommendations",
            "Rating Prediction",
            "About Project"
        ]
    )

    st.divider()

    st.caption("PEM FINAL PROJECT")

    st.caption("Python | Machine Learning | Streamlit")


# ==========================================
# 6. HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>✈️ Explore Incredible India</h1>
        <p>
        Discover tourist destinations across India
        through intelligent recommendations and
        machine learning insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Tourism Overview")

    total_places = len(clean_data)

    total_states = clean_data["State"].nunique()

    total_types = clean_data["Type"].nunique()

    avg_rating = clean_data["Google review rating"].mean()

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "🏛️ Total Destinations", total_places),
        (col2, "📍 States & UTs", total_states),
        (col3, "🎯 Attraction Types", total_types),
        (col4, "⭐ Average Rating", f"{avg_rating:.2f}/5")
    ]

    for column, label, value in metrics:

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # Tourism Charts

    st.subheader("Tourism Insights")

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("#### Top 10 Tourist States")

        state_counts = (
            clean_data["State"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        st.bar_chart(state_counts)

    with chart2:

        st.markdown("#### Popular Attraction Categories")

        category_counts = (
            clean_data["Type"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        st.bar_chart(category_counts)

    # Rating and Entrance Fee Analysis

    st.subheader("Attraction Analysis")

    chart3, chart4 = st.columns(2)

    with chart3:

        st.markdown("#### Rating Distribution")

        rating_counts = (
            clean_data["Google review rating"]
            .round(1)
            .value_counts()
            .sort_index()
        )

        st.bar_chart(rating_counts)

    with chart4:

        st.markdown("#### Free vs Paid Attractions")

        fee_counts = pd.Series({
            "Free Entry": (
                clean_data["Entrance Fee in INR"] == 0
            ).sum(),

            "Paid Entry": (
                clean_data["Entrance Fee in INR"] > 0
            ).sum()
        })

        st.bar_chart(fee_counts)

    # Top Rated Destinations

    st.subheader("Top Rated Tourist Destinations")

    top_places = (
        clean_data[
            [
                "Name",
                "State",
                "City",
                "Type",
                "Google review rating"
            ]
        ]
        .sort_values(
            "Google review rating",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
    )

    st.dataframe(
        top_places,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 7. DESTINATION RECOMMENDATIONS
# ==========================================

elif page == "Destination Recommendations":

    st.title("🔍 Destination Recommendations")

    st.write(
        "Select a tourist attraction to discover "
        "five similar destinations."
    )

    ml_data = recommendation_model["ml_data"]

    cosine_sim = recommendation_model["cosine_sim"]

    selected_place = st.selectbox(
        "Select Tourist Attraction",
        sorted(ml_data["Name"].unique())
    )

    if st.button("Find Similar Destinations"):

        place_index = ml_data[
            ml_data["Name"] == selected_place
        ].index[0]

        similarity_scores = list(
            enumerate(cosine_sim[place_index])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = [
            item for item in similarity_scores
            if item[0] != place_index
        ][:5]

        recommended_indices = [
            item[0] for item in similarity_scores
        ]

        recommendations = ml_data.iloc[
            recommended_indices
        ][
            ["Name", "State", "City", "Type", "Significance"]
        ].copy()

        recommendations["Similarity Score"] = [
            round(item[1], 3)
            for item in similarity_scores
        ]

        st.success("Recommendations generated successfully!")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 8. RATING PREDICTION
# ==========================================

elif page == "Rating Prediction":

    st.title("📊 Tourist Attraction Rating Prediction")

    st.write(
        "Predict the expected rating and rating category "
        "using your trained ML models."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            zone = st.selectbox(
                "Zone",
                sorted(clean_data["Zone"].dropna().unique())
            )

            state = st.selectbox(
                "State",
                sorted(clean_data["State"].dropna().unique())
            )

            city = st.selectbox(
                "City",
                sorted(clean_data["City"].dropna().unique())
            )

            attraction_type = st.selectbox(
                "Attraction Type",
                sorted(clean_data["Type"].dropna().unique())
            )

            significance = st.selectbox(
                "Significance",
                sorted(clean_data["Significance"].dropna().unique())
            )

        with col2:

            best_time = st.selectbox(
                "Best Time to Visit",
                sorted(clean_data["Best Time to visit"].dropna().unique())
            )

            entrance_fee = st.number_input(
                "Entrance Fee (INR)",
                min_value=0,
                value=0,
                step=50
            )

            visit_duration = st.number_input(
                "Visit Duration (Hours)",
                min_value=0.5,
                value=2.0,
                step=0.5
            )

            airport = st.selectbox(
                "Airport Within 50 km",
                sorted(clean_data["Airport with 50km Radius"].dropna().unique())
            )

            dslr = st.selectbox(
                "DSLR Allowed",
                sorted(clean_data["DSLR Allowed"].dropna().unique())
            )

        submitted = st.form_submit_button("Predict Rating")

    if submitted:

        input_data = pd.DataFrame([{
            "Zone": zone,
            "State": state,
            "City": city,
            "Type": attraction_type,
            "Significance": significance,
            "Best Time to visit": best_time,
            "Entrance Fee in INR": entrance_fee,
            "Time needed to visit in hrs": visit_duration,
            "Airport with 50km Radius": airport,
            "DSLR Allowed": dslr
        }])

        rf_model = prediction_models[
            "random_forest_regression"
        ]

        logistic_model = prediction_models[
            "logistic_regression"
        ]

        predicted_rating = rf_model.predict(input_data)[0]

        predicted_class = logistic_model.predict(input_data)[0]

        probabilities = logistic_model.predict_proba(input_data)[0]

        high_index = list(logistic_model.classes_).index(1)

        high_probability = probabilities[high_index] * 100

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Google Rating",
            f"{predicted_rating:.2f}/5"
        )

        col2.metric(
            "High Rating Probability",
            f"{high_probability:.1f}%"
        )

        if predicted_class == 1:

            st.success("Predicted Category: High Rating (4.5 or above)")

        else:

            st.warning("Predicted Category: Normal Rating (below 4.5)")

        st.caption(
            "Predictions are estimates based on the training dataset."
        )


# ==========================================
# 9. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Travel & Tourism Recommendation System")

    st.write(
        "This project uses machine learning to help users "
        "discover similar tourist destinations across India "
        "and estimate attraction ratings."
    )

    st.subheader("Algorithms Used")

    st.markdown("""
    **1. Content-Based Filtering**
    - TF-IDF Vectorization
    - Cosine Similarity

    **2. Random Forest Regression**
    - Predicts Google review ratings

    **3. Logistic Regression**
    - Classifies attractions into high-rating and normal-rating categories
    """)

    st.subheader("Technologies Used")

    st.write(
        "Python, Pandas, NumPy, Scikit-learn, "
        "Streamlit and Joblib"
    )

    st.subheader("Dataset")

    st.write(
        f"Cleaned dataset contains {len(clean_data)} tourist attractions."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Travel & Tourism Recommendation System | "
    "PEM Final Project | Machine Learning"
)
