"""
app.py
-------
Purpose:
    A professional Streamlit web application for the Iris Flower
    Classification project. It has 4 tabs:
      1. Dashboard   - KPIs, filters, and charts over the dataset
      2. EDA         - deeper exploratory charts (interactive, via Plotly)
      3. Predict     - a live prediction tool using the trained model
      4. Insights    - key findings & recommendations from the analysis

How to run this app:
    From the project's ROOT folder (the one containing "app/"), run:
        streamlit run app/app.py

Beginner note:
    Streamlit re-runs this ENTIRE script from top to bottom every time
    the user interacts with a widget (like moving a slider). That's
    why we use @st.cache_data / @st.cache_resource to avoid reloading
    the dataset or the model on every single interaction — it makes
    the app much faster.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit command in the script
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification | CodeAlpha",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# PATH SETUP
# We compute paths relative to this file so the app works no matter
# which folder you run `streamlit run` from.
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "iris_cleaned.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "models", "label_encoder.pkl")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "docs", "model_comparison_results.csv")


# ------------------------------------------------------------------
# CACHED LOADERS
# @st.cache_data caches DATA (DataFrames, arrays, etc.)
# @st.cache_resource caches OBJECTS that shouldn't be copied
# (like ML models, DB connections). Both avoid redoing slow work
# every time the user clicks something.
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, encoder, metadata


@st.cache_data
def load_model_results():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None


df = load_data()
model, label_encoder, metadata = load_model()
results_df = load_model_results()

SPECIES_EMOJI = {"setosa": "🌱", "versicolor": "🌺", "virginica": "🌷"}
SPECIES_COLOR = {"setosa": "#2ecc71", "versicolor": "#e67e22", "virginica": "#8e44ad"}


# ------------------------------------------------------------------
# CUSTOM CSS — a bit of styling to make the dashboard look polished
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: -10px;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px;
    }
    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<p class="main-header">🌸 Iris Flower Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">An end-to-end Machine Learning project — CodeAlpha Data Science Internship (Task 1)</p>',
    unsafe_allow_html=True,
)
st.divider()


# ------------------------------------------------------------------
# SIDEBAR — FILTERS
# ------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

species_options = sorted(df["species"].unique().tolist())
selected_species = st.sidebar.multiselect(
    "Filter by species",
    options=species_options,
    default=species_options,
)

sepal_len_range = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df["sepal_length_cm"].min()), float(df["sepal_length_cm"].max()),
    (float(df["sepal_length_cm"].min()), float(df["sepal_length_cm"].max())),
)
petal_len_range = st.sidebar.slider(
    "Petal Length (cm)",
    float(df["petal_length_cm"].min()), float(df["petal_length_cm"].max()),
    (float(df["petal_length_cm"].min()), float(df["petal_length_cm"].max())),
)

# Apply the filters chosen by the user in the sidebar
filtered_df = df[
    (df["species"].isin(selected_species)) &
    (df["sepal_length_cm"].between(*sepal_len_range)) &
    (df["petal_length_cm"].between(*petal_len_range))
]

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This dashboard explores the classic Iris dataset (Fisher, 1936) "
    "and lets you classify new flower measurements using a trained "
    f"**{metadata['best_model_name']}** model "
    f"({metadata['test_accuracy']*100:.1f}% test accuracy)."
)
st.sidebar.markdown("[View source on GitHub](#)")


# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 EDA Explorer", "🤖 Predict", "💡 Insights"])


# ====================================================================
# TAB 1: DASHBOARD (KPIs + filtered charts)
# ====================================================================
with tab1:
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples (filtered)", f"{len(filtered_df)}")
    col2.metric("Species Shown", f"{filtered_df['species'].nunique()}")
    col3.metric("Avg Petal Length", f"{filtered_df['petal_length_cm'].mean():.2f} cm" if len(filtered_df) else "N/A")
    col4.metric("Best Model Accuracy", f"{metadata['test_accuracy']*100:.1f}%")

    st.markdown("### ")

    if len(filtered_df) == 0:
        st.warning("No data matches the current filters. Try widening the sliders in the sidebar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Species Distribution**")
            species_counts = filtered_df["species"].value_counts().reset_index()
            species_counts.columns = ["species", "count"]
            fig = px.bar(
                species_counts, x="species", y="count", color="species",
                color_discrete_map=SPECIES_COLOR, text="count",
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Petal Length vs Petal Width**")
            fig = px.scatter(
                filtered_df, x="petal_length_cm", y="petal_width_cm", color="species",
                color_discrete_map=SPECIES_COLOR, size_max=10,
                labels={"petal_length_cm": "Petal Length (cm)", "petal_width_cm": "Petal Width (cm)"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**Feature Averages by Species**")
            avg_df = filtered_df.groupby("species")[
                ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
            ].mean().reset_index()
            avg_melted = avg_df.melt(id_vars="species", var_name="feature", value_name="value")
            fig = px.bar(
                avg_melted, x="feature", y="value", color="species", barmode="group",
                color_discrete_map=SPECIES_COLOR,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.markdown("**Sepal Length Distribution**")
            fig = px.histogram(
                filtered_df, x="sepal_length_cm", color="species", marginal="box",
                color_discrete_map=SPECIES_COLOR, opacity=0.75,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Filtered Data")
        st.dataframe(filtered_df, use_container_width=True, height=250)

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered data as CSV", csv, "filtered_iris_data.csv", "text/csv")


# ====================================================================
# TAB 2: EDA EXPLORER
# ====================================================================
with tab2:
    st.subheader("Exploratory Data Analysis")

    eda_metric = st.selectbox(
        "Choose a feature to explore",
        ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"],
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Boxplot of {eda_metric} by species**")
        fig = px.box(df, x="species", y=eda_metric, color="species", color_discrete_map=SPECIES_COLOR)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**Violin plot of {eda_metric} by species**")
        fig = px.violin(df, x="species", y=eda_metric, color="species", box=True,
                         color_discrete_map=SPECIES_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Correlation Heatmap (all numeric features)**")
    numeric_df = df.drop(columns=["species_id", "species"], errors="ignore")
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**3D Scatter — Sepal Length, Petal Length, Petal Width**")
    fig = px.scatter_3d(
        df, x="sepal_length_cm", y="petal_length_cm", z="petal_width_cm",
        color="species", color_discrete_map=SPECIES_COLOR,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Descriptive Statistics by Species**")
    st.dataframe(df.groupby("species").describe().T, use_container_width=True)


# ====================================================================
# TAB 3: PREDICT (live model inference)
# ====================================================================
with tab3:
    st.subheader("🤖 Classify a New Iris Flower")
    st.write("Move the sliders to enter flower measurements, then click **Predict Species**.")

    pc1, pc2 = st.columns([1, 1.2])

    with pc1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.6, 1.2, 0.1)

        predict_btn = st.button("🔮 Predict Species", type="primary", use_container_width=True)

    with pc2:
        if predict_btn:
            # Build a single-row DataFrame matching the training feature columns/order
            input_df = pd.DataFrame([{
                "sepal_length_cm": sepal_length,
                "sepal_width_cm": sepal_width,
                "petal_length_cm": petal_length,
                "petal_width_cm": petal_width,
            }])[metadata["feature_columns"]]

            pred_encoded = model.predict(input_df)[0]
            pred_species = label_encoder.inverse_transform([pred_encoded])[0]

            # predict_proba gives the model's confidence for each class
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
            else:
                probabilities = None

            emoji = SPECIES_EMOJI.get(pred_species, "🌸")
            st.success(f"### {emoji} Predicted Species: **{pred_species.capitalize()}**")

            if probabilities is not None:
                prob_df = pd.DataFrame({
                    "Species": label_encoder.classes_,
                    "Confidence": probabilities,
                }).sort_values("Confidence", ascending=False)

                fig = px.bar(
                    prob_df, x="Confidence", y="Species", orientation="h",
                    color="Species", color_discrete_map=SPECIES_COLOR,
                    text=prob_df["Confidence"].apply(lambda x: f"{x*100:.1f}%"),
                )
                fig.update_layout(showlegend=False, height=280, xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Your input values:**")
            st.dataframe(input_df, use_container_width=True)
        else:
            st.info("👈 Set your measurements and click **Predict Species** to see the result here.")

    st.divider()
    st.markdown("### Try These Example Flowers")
    examples = {
        "Typical Setosa 🌱": (5.1, 3.5, 1.4, 0.2),
        "Typical Versicolor 🌺": (6.0, 2.7, 4.2, 1.3),
        "Typical Virginica 🌷": (6.5, 3.0, 5.5, 2.0),
    }
    ex_cols = st.columns(3)
    for i, (label, values) in enumerate(examples.items()):
        with ex_cols[i]:
            st.markdown(f"**{label}**")
            st.caption(f"Sepal: {values[0]} x {values[1]} cm | Petal: {values[2]} x {values[3]} cm")


# ====================================================================
# TAB 4: INSIGHTS & RECOMMENDATIONS
# ====================================================================
with tab4:
    st.subheader("💡 Key Insights from the Analysis")

    st.markdown("""
    1. **Petal measurements are far better predictors than sepal measurements.**
       Petal length and petal width alone almost perfectly separate the three
       species — this is visible in the scatter plot and confirmed by feature
       importance from the tree-based models.

    2. **Setosa is the easiest species to identify.**
       It is linearly separable from the other two species using petal length
       alone (its petals are consistently much smaller). No model ever confuses
       Setosa with another species in this dataset.

    3. **Versicolor and Virginica have some overlap.**
       These two species are the main source of any classification errors,
       since their petal and sepal measurements overlap slightly at the
       boundary. This is where model choice matters most.

    4. **The dataset is small but very clean.**
       After cleaning, 149 samples remained (1 duplicate removed), with only
       a handful of mild outliers in sepal width — well within normal
       biological variation.
    """)

    if results_df is not None:
        st.markdown("### 🏆 Model Comparison Results")
        st.dataframe(
            results_df.style.background_gradient(cmap="Greens", subset=["Test Accuracy", "CV Mean Accuracy"]),
            use_container_width=True,
        )
        st.caption(
            f"**Best model: {metadata['best_model_name']}** — selected for the highest test accuracy "
            f"({metadata['test_accuracy']*100:.1f}%) combined with strong, stable cross-validation performance."
        )

    st.markdown("### 📋 Recommendations")
    st.markdown("""
    - **For production use:** prioritize collecting accurate **petal length and
      petal width** measurements — sepal measurements add comparatively little
      predictive value and could be dropped to simplify data collection.
    - **For further accuracy gains:** collect more samples, especially near the
      Versicolor/Virginica boundary, to help the model learn a sharper decision
      boundary between these two species.
    - **For deployment:** the trained model is small and fast (SVM with 4
      input features) and is well suited for real-time or edge deployment,
      e.g. inside a mobile field-data-collection app for botanists.
    """)

st.markdown('<p class="footer-note">Built with ❤️ using Python, scikit-learn & Streamlit — CodeAlpha Data Science Internship</p>', unsafe_allow_html=True)
