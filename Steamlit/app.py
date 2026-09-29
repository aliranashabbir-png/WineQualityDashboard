from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import plotly.express as px
except Exception:
    px = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "WineQT.csv"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "quality"

QUALITY_LABELS = {
    3: "Low",
    4: "Low",
    5: "Average",
    6: "Good",
    7: "Very Good",
    8: "Excellent",
}


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def build_model():
    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return model, accuracy, report


st.set_page_config(
    page_title="Wine Quality Dashboard",
    page_icon="🍷",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #120d16 0%, #111827 28%, #1a2236 62%, #0f172a 100%);
    }
    .stApp {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.28);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 12px 32px rgba(2, 6, 23, 0.35);
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(76,29,149,0.32), rgba(59,130,246,0.14), rgba(168,85,247,0.18));
        border: 1px solid rgba(196,181,253,0.42);
        border-radius: 22px;
        padding: 1.5rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 18px 35px rgba(15, 23, 42, 0.25);
    }
    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: linear-gradient(135deg, #f59e0b, #f43f5e);
        color: white;
        font-size: 1.7rem;
        font-weight: 800;
        box-shadow: 0 10px 20px rgba(244, 63, 94, 0.25);
    }
    .brand-name {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.04rem;
        color: #ffffff;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(34, 197, 94, 0.18);
        color: #dcfce7;
        border: 1px solid rgba(34, 197, 94, 0.4);
        letter-spacing: 0.04rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.75rem;
    }
    .mini-brand {
        font-size: 0.82rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        color: #cbd5e1;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


df = load_dataset()
model, accuracy, report = build_model()

quality_counts = df[TARGET].value_counts().sort_index()
median_values = df[FEATURES].median().to_dict()
correlation_df = df[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)
correlation_df = correlation_df.drop(labels=TARGET, errors="ignore")
feature_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

st.markdown(
    """
    <div class="hero-box">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;">
            <div style="display:flex; align-items:center; gap:18px; flex-wrap:wrap;">
                <div class="brand-mark">V</div>
                <div>
                    <div class="mini-brand">VineSense AI</div>
                    <div class="badge">PRESENTATION VERSION</div>
                    <h1 style="margin: 0.7rem 0 0.25rem 0; font-size: 2.5rem; line-height: 1.15;">Wine Quality Prediction Dashboard</h1>
                    <p style="margin: 0; color: #dbeafe; max-width: 760px;">A premium AI analytics experience for classifying wine quality using chemistry, signal strength, and model-driven insights.</p>
                </div>
            </div>
            <div style="text-align:right; min-width: 180px;">
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08rem; color: #cbd5e1;">Model</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #ffffff;">Random Forest</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Dataset rows", f"{len(df):,}")
metric_cols[1].metric("Average quality", f"{df[TARGET].mean():.2f}")
metric_cols[2].metric("Model accuracy", f"{accuracy * 100:.2f}%")
metric_cols[3].metric("Quality range", "3 - 8")

st.markdown("---")

overview_tab, predict_tab, insights_tab = st.tabs(["Overview", "Prediction", "Insights"])

with overview_tab:
    left, right = st.columns([1.6, 1])
    with left:
        st.markdown('<div class="section-title">Quality distribution</div>', unsafe_allow_html=True)
        st.bar_chart(quality_counts)

    with right:
        st.markdown('<div class="section-title">Key quality signals</div>', unsafe_allow_html=True)
        top_signals = correlation_df.head(5)
        for feature, value in top_signals.items():
            st.write(f"- {feature.title()}: {value:.3f}")

    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown('<div class="section-title">Quality mix</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame({
            "Quality": quality_counts.index.astype(str),
            "Count": quality_counts.values,
        })
        if px is not None:
            fig = px.pie(
                pie_df,
                names="Quality",
                values="Count",
                hole=0.45,
                color_discrete_sequence=["#f59e0b", "#f472b6", "#60a5fa", "#34d399", "#a78bfa", "#f87171"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc"},
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.write("Pie chart unavailable")

    with chart2:
        st.markdown('<div class="section-title">Feature contribution overview</div>', unsafe_allow_html=True)
        importance_df = feature_importance.reset_index()
        importance_df.columns = ["Feature", "Importance"]
        if px is not None:
            fig2 = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc"},
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.bar_chart(feature_importance)

    st.markdown("### Executive summary")
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.dataframe(df.head(10), width="stretch")
    with summary_right:
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df, width="stretch")

with predict_tab:
    sidebar = st.sidebar
    sidebar.header("🍇 Wine chemistry inputs")
    sidebar.caption("Adjust the values below to simulate a wine profile and generate a quality prediction.")

    user_inputs = {}
    for feature in FEATURES:
        user_inputs[feature] = sidebar.number_input(
            feature.title(),
            min_value=0.0,
            value=float(median_values[feature]),
            step=0.01,
            format="%.2f",
        )

    sidebar.markdown("---")
    sample_button = sidebar.button("Load sample wine")
    if sample_button:
        user_inputs = {
            "fixed acidity": 7.4,
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        }

    predict_button = sidebar.button("Predict quality", type="primary")

    if predict_button:
        input_df = pd.DataFrame([user_inputs], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100
        quality_label = QUALITY_LABELS.get(prediction, "Custom")

        result_col, details_col = st.columns([1.2, 1])
        with result_col:
            st.markdown(
                f"""
                <div class="glass-card" style="background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(17,24,39,0.8)); border: 1px solid rgba(34,197,94,0.52); padding: 1.3rem; border-radius: 18px;">
                    <h3 style="margin: 0; color: #dcfce7;">Predicted wine quality</h3>
                    <h1 style="margin: 0.4rem 0; font-size: 3rem; color: #ffffff;">{prediction}</h1>
                    <p style="margin: 0; color: #bbf7d0;">Category: {quality_label}</p>
                    <p style="margin-top: 0.8rem; margin-bottom: 0; color: #bef264;">Confidence: {confidence:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            guide = {
                "Low": "Quality 3–4: more acidic and less balanced profile.",
                "Average": "Quality 5: balanced but not premium.",
                "Good": "Quality 6: strong overall balance and marketable taste.",
                "Very Good": "Quality 7: premium characteristics and better structure.",
                "Excellent": "Quality 8: top-tier wine profile and very high quality.",
            }
            st.info(guide.get(quality_label, "This wine falls outside the usual quality ranges."))

        with details_col:
            st.write("### Current input values")
            st.json(user_inputs)

        st.markdown("---")
        st.write("### Model confidence")
        prob_df = pd.DataFrame({
            "Class": model.classes_.astype(str),
            "Probability": probabilities * 100,
        })
        st.bar_chart(prob_df.set_index("Class"))

    else:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Ready to predict</h3>
                <p>Use the left panel to enter wine chemistry values or load a sample profile to test the model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with insights_tab:
    st.subheader("Top feature importance")
    st.bar_chart(feature_importance)

    st.subheader("Strongest feature correlations with quality")
    corr_df = pd.DataFrame({
        "Feature": correlation_df.index,
        "Correlation with Quality": correlation_df.values,
    })
    st.dataframe(corr_df, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Smart observation")
        st.success("Alcohol and volatile acidity are usually among the strongest quality signals in this dataset.")
    with col2:
        st.markdown("### Model note")
        st.info("The model is built to classify wine quality from measurable chemistry, not from subjective tasting notes or visual inspection.")

    st.subheader("Dataset overview")
    st.dataframe(df.describe().T, width="stretch")

    st.markdown("---")
    st.markdown("### Realistic interpretation")
    st.write("- Most wines in the dataset fall into medium quality categories, which is typical for real-world wine collection and production data.")
    st.write("- Features such as alcohol, acidity, and sulphates appear to influence perceived quality more than the other variables.")
    st.write("- The model provides a practical decision-support tool for wine evaluation, especially when tasting notes are unavailable.")
st.caption("Run this app from the project root with: streamlit run Steamlit/app.py")