from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from anomalyx.pipeline import DEFAULT_CONTAMINATION, PipelineValidationError, prepare_data, train_isolation_forest

st.set_page_config(page_title="AnomalyX Lab", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

APP_ROOT = Path(__file__).parent
SAMPLE_PATH = APP_ROOT / "data" / "anomalyx_sample.csv"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink: #edf2ff; --muted: #8792b0; --line: #26304a; --panel: #11182b; --bg: #080d1c; --cyan: #61e7dc; --violet: #9b8cff; --rose: #ff719f; }
        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
        .stApp { background: radial-gradient(circle at 80% 0%, #1c1842 0, #080d1c 34rem), var(--bg); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #0b1223; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
        .block-container { padding: 2.2rem 3.2rem 4rem; max-width: 1500px; }
        h1, h2, h3 { letter-spacing: -0.04em; }
        h1 { font-size: clamp(2.4rem, 5vw, 4.7rem) !important; line-height: .98 !important; margin: .6rem 0 .8rem !important; }
        h2 { font-size: 1.45rem !important; margin-top: 1.8rem !important; }
        h3 { font-size: 1.05rem !important; }
        p, label, [data-testid="stMarkdownContainer"] { color: var(--muted); }
        .eyebrow { color: var(--cyan); font-family: 'DM Mono', monospace; font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; }
        .hero-copy { color: #aab5d2; font-size: 1.05rem; max-width: 650px; line-height: 1.7; }
        .hero-mark { color: var(--cyan); font-family: 'DM Mono', monospace; font-size: 1.05rem; }
        .metric-card { background: linear-gradient(145deg, rgba(24,35,61,.94), rgba(13,19,36,.94)); border: 1px solid var(--line); border-radius: 18px; padding: 1.15rem 1.25rem; min-height: 112px; }
        .metric-label { font-family: 'DM Mono', monospace; color: var(--muted); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; }
        .metric-value { color: var(--ink); font-size: 2rem; font-weight: 600; margin-top: .45rem; }
        .metric-accent { color: var(--rose); }
        .status-pill { display: inline-block; color: var(--cyan); border: 1px solid rgba(97,231,220,.35); background: rgba(97,231,220,.08); border-radius: 999px; padding: .28rem .65rem; font: .72rem 'DM Mono', monospace; }
        .section-kicker { color: var(--violet); font: .72rem 'DM Mono', monospace; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .45rem; }
        .note { background: rgba(155,140,255,.08); border-left: 2px solid var(--violet); padding: .8rem 1rem; border-radius: 0 10px 10px 0; color: #b8b6dd; line-height: 1.55; font-size: .9rem; }
        .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 1px solid var(--line); font-weight: 600; }
        .stButton > button[kind="primary"] { background: linear-gradient(90deg, #6b5eff, #8a7dff); border: 0; }
        div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section { background: #121a2e; border-color: var(--line); }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .small-mono { font: .72rem 'DM Mono', monospace; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def read_upload(uploaded_file) -> pd.DataFrame:
    try:
        raw = uploaded_file.getvalue()
        if not raw:
            raise PipelineValidationError("The uploaded file is empty.")
        return pd.read_csv(io.BytesIO(raw))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise PipelineValidationError(f"Could not parse this CSV: {exc}") from exc


def metric(label: str, value: str, accent: bool = False) -> None:
    accent_class = " metric-accent" if accent else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value{accent_class}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def plot_results(results: pd.DataFrame, features: list[str]) -> None:
    x_feature = features[0]
    y_feature = features[1] if len(features) > 1 else features[0]
    plot_frame = results.copy()
    plot_frame["row_id"] = np.arange(1, len(plot_frame) + 1)
    if x_feature == y_feature:
        fig = px.strip(
            plot_frame,
            x=x_feature,
            color="prediction",
            hover_data=["row_id", "anomaly_score"],
            color_discrete_map={"Normal": "#61e7dc", "Anomaly": "#ff719f"},
        )
    else:
        fig = px.scatter(
            plot_frame,
            x=x_feature,
            y=y_feature,
            color="prediction",
            symbol="prediction",
            hover_data=["row_id", "anomaly_score"],
            color_discrete_map={"Normal": "#61e7dc", "Anomaly": "#ff719f"},
        )
    fig.update_traces(marker={"size": 9, "line": {"width": 1, "color": "#080d1c"}})
    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,43,.6)",
        legend_title_text="",
        font={"family": "Space Grotesk", "color": "#aab5d2"},
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def main() -> None:
    apply_theme()
    with st.sidebar:
        st.markdown('<div class="hero-mark">◈ ANOMALYX</div>', unsafe_allow_html=True)
        st.markdown("### Input console")
        source = st.radio("Data source", ["Built-in sample", "Upload CSV"], help="Use the sample to explore the full lab without preparing a file.")
        uploaded_file = st.file_uploader("Drop a CSV file", type=["csv"], help="CSV files should contain at least 8 rows and one numeric column.")
        if source == "Upload CSV":
            if uploaded_file is not None:
                try:
                    frame = read_upload(uploaded_file)
                    source_label = uploaded_file.name
                except PipelineValidationError as exc:
                    st.error(str(exc))
                    frame = None
                    source_label = "Invalid CSV"
            else:
                frame = None
                source_label = "No CSV uploaded"
        else:
            frame = load_sample()
            source_label = "Built-in sample · synthetic"
            st.caption("The built-in dataset is synthetic demo data, not real-world evidence.")
        st.divider()
        st.markdown('<div class="section-kicker">Model controls</div>', unsafe_allow_html=True)
        contamination = st.slider(
            "Expected anomaly rate",
            min_value=0.01,
            max_value=0.30,
            value=DEFAULT_CONTAMINATION,
            step=0.01,
            format="%0.2f",
            help="Isolation Forest uses this as the expected proportion of outliers. It is a modeling assumption, not a measured truth.",
        )
        run_model = st.button("Run anomaly scan", type="primary", width="stretch", disabled=frame is None)
        st.markdown('<div class="note">Scores rank how isolated a row is from the rest of the selected data. They are not probabilities or proof of fraud.</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Interactive machine learning lab · reproducible scan</div>', unsafe_allow_html=True)
    st.title("Find the unusual\nwithout losing the context.")
    st.markdown('<div class="hero-copy">AnomalyX turns a table into an explorable anomaly map. Select the signals that matter, tune the expected anomaly rate, and inspect every flagged row.</div>', unsafe_allow_html=True)
    st.write("")
    if frame is None:
        st.warning("Choose the built-in sample or upload a valid CSV to begin.")
        return

    try:
        default_features = frame.select_dtypes(include=[np.number]).columns.tolist()
        prepared = prepare_data(frame, default_features[:12])
    except PipelineValidationError as exc:
        st.error(str(exc))
        return

    st.markdown(f'<span class="status-pill">SOURCE · {source_label}</span> &nbsp; <span class="small-mono">{len(frame):,} rows · {len(frame.columns)} columns</span>', unsafe_allow_html=True)
    st.subheader("1. Inspect your data")
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(frame.head(8), width="stretch", hide_index=True)
    with right:
        chosen_features = st.multiselect(
            "Numerical features",
            options=prepared.numeric_candidates,
            default=prepared.features,
            max_selections=12,
            help="Isolation Forest works on numeric signals. Non-numeric columns remain in the downloadable results but are not used for training.",
        )
        missing_total = int(frame.isna().sum().sum())
        st.metric("Missing cells in source", f"{missing_total:,}", help="Missing numeric values are median-imputed during training. Rows are retained in the output.")
        if missing_total:
            missing_table = frame.isna().sum().rename("missing").to_frame()
            missing_table["rate"] = (missing_table["missing"] / len(frame)).round(4)
            st.dataframe(missing_table[missing_table["missing"] > 0], width="stretch", hide_index=False)
        else:
            st.success("No missing values detected in the source table.")

    if chosen_features != prepared.features:
        try:
            prepared = prepare_data(frame, chosen_features)
        except PipelineValidationError as exc:
            st.error(str(exc))
            return

    data_signature = (
        source_label,
        frame.shape,
        tuple(frame.columns),
        int(pd.util.hash_pandas_object(frame, index=True).sum()),
        tuple(prepared.features),
    )
    has_current_results = st.session_state.get("data_signature") == data_signature
    has_current_contamination = st.session_state.get("contamination") == contamination
    if run_model or "results" not in st.session_state:
        try:
            with st.spinner("Fitting Isolation Forest…"):
                results, model = train_isolation_forest(prepared, contamination)
            st.session_state.results = results
            st.session_state.model_features = prepared.features
            st.session_state.contamination = contamination
            st.session_state.data_signature = data_signature
        except PipelineValidationError as exc:
            st.error(str(exc))
            return

    if not has_current_results or not has_current_contamination:
        st.info("Your data, features, or model setting changed. Click Run anomaly scan to refresh the results.")
        return
    results = st.session_state.results
    anomalies = int(results["is_anomaly"].sum())
    st.subheader("2. Scan summary")
    cards = st.columns(4)
    with cards[0]: metric("Total records", f"{len(results):,}")
    with cards[1]: metric("Anomalies detected", f"{anomalies:,}", True)
    with cards[2]: metric("Anomaly percentage", f"{anomalies / len(results):.1%}", True)
    with cards[3]: metric("Features in model", f"{len(prepared.features):,}")
    st.subheader("3. Explore the anomaly map")
    chart_col, distribution_col = st.columns([1.45, 1])
    with chart_col:
        st.caption("Anomalies are highlighted in coral. Hover over any point to inspect its row ID and score.")
        plot_results(results, prepared.features)
    with distribution_col:
        distribution_feature = st.selectbox("Feature distribution", prepared.features, help="Compare the selected feature for normal and flagged rows.")
        distribution_fig = px.histogram(
            results,
            x=distribution_feature,
            color="prediction",
            marginal="box",
            nbins=28,
            opacity=.78,
            color_discrete_map={"Normal": "#61e7dc", "Anomaly": "#ff719f"},
        )
        distribution_fig.update_layout(template="plotly_dark", height=460, margin={"l": 10, "r": 10, "t": 20, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,43,.6)", legend_title_text="", font={"family": "Space Grotesk", "color": "#aab5d2"})
        st.plotly_chart(distribution_fig, width="stretch", config={"displaylogo": False})
    st.subheader("4. Review and export")
    visible_columns = [column for column in results.columns if column not in ["is_anomaly"]]
    st.dataframe(results[visible_columns].sort_values("anomaly_score", ascending=False).head(25), width="stretch", hide_index=True)
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results CSV", data=csv_bytes, file_name="anomalyx_results.csv", mime="text/csv", type="primary")
    st.markdown('<div class="note">How to read the score: larger anomaly scores indicate greater isolation relative to this dataset. The cutoff is influenced by contamination and the selected features. A flagged record is a candidate for review, not a confirmed incident or fraudulent transaction.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
