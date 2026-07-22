"""
app.py
-------
Purpose:
    Streamlit dashboard for the "Unemployment Analysis with Python"
    project (CodeAlpha Task 2). 4 tabs: Dashboard, COVID Impact,
    Predict, Insights.

How to run:
    From the project ROOT folder, run:
        streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px

st.set_page_config(
    page_title="Unemployment Analysis India | CodeAlpha",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "unemployment_features.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "docs", "model_comparison_results.csv")

LOCKDOWN_DATE = pd.Timestamp("2020-03-25")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data
def load_model_results():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None


df = load_data()
model, metadata = load_model()
results_df = load_model_results()

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ee0979 0%, #ff6a00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { font-size: 1.05rem; color: #6b7280; margin-top: -10px; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 15px;
    }
    .footer-note { text-align: center; color: #9ca3af; font-size: 0.85rem; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📉 Unemployment in India — Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Exploring the impact of COVID-19 on employment — CodeAlpha Data Science Internship (Task 2)</p>',
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------------------
st.sidebar.header("🔎 Filters")

state_options = sorted(df["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Filter by state", options=state_options, default=state_options)

date_min, date_max = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.slider("Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max))

area_options = ["Rural", "Urban", "Combined"]
selected_areas = st.sidebar.multiselect("Filter by area", options=area_options, default=area_options)

filtered_df = df[
    (df["state"].isin(selected_states)) &
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1]) &
    (df["area_filled"].isin(selected_areas))
]

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This dashboard analyzes unemployment trends across Indian states "
    "before and during the COVID-19 lockdown, and predicts unemployment "
    f"rate using a trained **{metadata['best_model_name']}** model "
    f"(R² = {metadata['r2_score']:.2f})."
)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🦠 COVID Impact", "🤖 Predict", "💡 Insights"])

# ====================================================================
# TAB 1: DASHBOARD
# ====================================================================
with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records Shown", f"{len(filtered_df)}")
    col2.metric("States Shown", f"{filtered_df['state'].nunique()}")
    col3.metric("Avg Unemployment Rate", f"{filtered_df['unemployment_rate'].mean():.2f}%" if len(filtered_df) else "N/A")
    col4.metric("Model R² Score", f"{metadata['r2_score']:.2f}")

    if len(filtered_df) == 0:
        st.warning("No data matches the current filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Unemployment Rate Over Time**")
            monthly = filtered_df.groupby("date")["unemployment_rate"].mean().reset_index()
            fig = px.line(monthly, x="date", y="unemployment_rate", markers=True)
            fig.add_vline(x=LOCKDOWN_DATE, line_dash="dash", line_color="black")
            fig.update_layout(height=380, yaxis_title="Unemployment Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Top 10 States by Avg Unemployment Rate**")
            top10 = filtered_df.groupby("state")["unemployment_rate"].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top10, x="unemployment_rate", y="state", orientation="h", color="unemployment_rate",
                         color_continuous_scale="Reds")
            fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Urban vs Rural Comparison**")
            area_df = filtered_df.dropna(subset=["area"])
            if len(area_df):
                monthly_area = area_df.groupby(["date", "area"])["unemployment_rate"].mean().reset_index()
                fig = px.line(monthly_area, x="date", y="unemployment_rate", color="area", markers=True)
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Urban/Rural split data in current filter selection.")

        with c4:
            st.markdown("**Labour Participation Rate Over Time**")
            monthly_lpr = filtered_df.groupby("date")["labour_participation_rate"].mean().reset_index()
            fig = px.line(monthly_lpr, x="date", y="labour_participation_rate", markers=True,
                         color_discrete_sequence=["#9b59b6"])
            fig.update_layout(height=380, yaxis_title="Labour Participation Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Filtered Data")
        st.dataframe(filtered_df, use_container_width=True, height=250)
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered data as CSV", csv, "filtered_unemployment_data.csv", "text/csv")


# ====================================================================
# TAB 2: COVID IMPACT
# ====================================================================
with tab2:
    st.subheader("🦠 COVID-19 Lockdown Impact Analysis")

    before = df[df["date"] < LOCKDOWN_DATE]["unemployment_rate"].mean()
    during = df[df["date"] >= LOCKDOWN_DATE]["unemployment_rate"].mean()
    increase_pct = (during / before - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Unemployment BEFORE Lockdown", f"{before:.2f}%")
    c2.metric("Avg Unemployment DURING/AFTER Lockdown", f"{during:.2f}%", delta=f"{during-before:.2f} pts")
    c3.metric("Relative Increase", f"{increase_pct:.1f}%")

    df_period = df.copy()
    df_period["period"] = df_period["date"].apply(lambda d: "Before Lockdown" if d < LOCKDOWN_DATE else "During/After Lockdown")
    fig = px.box(df_period, x="period", y="unemployment_rate", color="period",
                 color_discrete_map={"Before Lockdown": "#95a5a6", "During/After Lockdown": "#e74c3c"})
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Zone-wise Average Unemployment Rate (2020 data)**")
    zone_df = df.dropna(subset=["zone"])
    if len(zone_df):
        avg_zone = zone_df.groupby("zone")["unemployment_rate"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_zone, x="zone", y="unemployment_rate", color="zone")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**State x Month Heatmap**")
    pivot = df.pivot_table(index="state", columns="date", values="unemployment_rate", aggfunc="mean")
    pivot.columns = [d.strftime("%b-%y") for d in pivot.columns]
    fig = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                     labels=dict(color="Unemployment Rate (%)"))
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# TAB 3: PREDICT
# ====================================================================
with tab3:
    st.subheader("🤖 Predict Unemployment Rate")
    st.write("Enter the indicators below to estimate the unemployment rate.")

    pc1, pc2 = st.columns([1, 1.2])

    state_avg_lookup = df.groupby("state")["state_avg_unemployment"].first().to_dict()

    with pc1:
        selected_state = st.selectbox("State", options=state_options)
        lpr = st.slider("Labour Participation Rate (%)", 15.0, 70.0, 41.0, 0.1)
        month = st.selectbox("Month", options=list(range(1, 13)), format_func=lambda m: pd.Timestamp(2020, m, 1).strftime("%B"))
        year = st.selectbox("Year", options=[2019, 2020, 2021, 2022])
        lockdown = st.radio("Period", options=["Before Lockdown", "During/After Lockdown"])

        predict_btn = st.button("🔮 Predict Unemployment Rate", type="primary", use_container_width=True)

    with pc2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "labour_participation_rate": lpr,
                "month": month,
                "year": year,
                "is_lockdown_period": 1 if lockdown == "During/After Lockdown" else 0,
                "state_avg_unemployment": state_avg_lookup.get(selected_state, df["state_avg_unemployment"].mean()),
            }])[metadata["feature_columns"]]

            prediction = model.predict(input_df)[0]

            st.success(f"### 📊 Predicted Unemployment Rate: **{prediction:.2f}%**")

            fig = px.bar(x=[selected_state], y=[prediction], color=[selected_state],
                         labels={"x": "State", "y": "Predicted Unemployment Rate (%)"})
            fig.update_layout(showlegend=False, height=300, yaxis_range=[0, max(50, prediction + 10)])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Your input values:**")
            st.dataframe(input_df, use_container_width=True)
        else:
            st.info("👈 Set the inputs and click **Predict Unemployment Rate** to see the result.")


# ====================================================================
# TAB 4: INSIGHTS
# ====================================================================
with tab4:
    st.subheader("💡 Key Insights from the Analysis")

    st.markdown(f"""
    1. **COVID-19 lockdown sharply increased unemployment.** The national
       average unemployment rate rose from **{before:.2f}%** before the
       25 March 2020 lockdown to **{during:.2f}%** during/after it — a
       **{increase_pct:.1f}% relative increase**.

    2. **The effect was not evenly distributed.** Some states saw far
       sharper spikes than others (see the Top 10 States chart and the
       state x month heatmap) — states with a larger informal/urban labor
       force were generally hit hardest.

    3. **Labour force participation also dropped** during the lockdown
       period, meaning fewer people were even actively seeking work,
       which likely understates the true economic impact if participation
       had stayed constant.

    4. **Urban and rural areas were both affected**, but the magnitude and
       timing of the spike differed between them (see the Urban vs Rural
       chart in the Dashboard tab).
    """)

    if results_df is not None:
        st.markdown("### 🏆 Model Comparison Results")
        st.dataframe(
            results_df.style.background_gradient(cmap="Greens", subset=["R2 Score", "CV Mean R2"]),
            use_container_width=True,
        )
        st.caption(
            f"**Best model: {metadata['best_model_name']}** — R² = {metadata['r2_score']:.2f}, "
            f"MAE = {metadata['mae']:.2f} percentage points."
        )

    st.markdown("### 📋 Recommendations")
    st.markdown("""
    - **For policymakers:** target relief and job-creation programs at the
      states and periods showing the sharpest unemployment spikes, rather
      than applying uniform national measures.
    - **For future crisis preparedness:** the strong link between the
      lockdown period and unemployment rate suggests economic contingency
      plans should be triggered proactively alongside public health
      measures, not after the fact.
    - **For further analysis:** incorporating sector-level data (which
      industries lost the most jobs) would sharpen where interventions
      are needed most.
    """)

st.markdown('<p class="footer-note">Built with ❤️ using Python, scikit-learn & Streamlit — CodeAlpha Data Science Internship</p>', unsafe_allow_html=True)
