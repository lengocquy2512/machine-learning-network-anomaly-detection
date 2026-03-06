import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils.model_io import load_best_model
from utils.data_prep import prepare_for_best_model
from utils.metrics_ext import ids_metrics


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


# -----------------------------
# UI helpers
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Layout */
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
        section[data-testid="stSidebar"] {width: 320px !important;}

        /* Hide Streamlit default */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Typography */
        h1, h2, h3 {letter-spacing: -0.02em;}
        .muted {opacity: 0.75; font-size: 0.92rem;}

        /* Card style */
        .card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.03);
        }
        .card-title {font-size: 0.9rem; opacity: 0.8; margin-bottom: 6px;}
        .card-value {font-size: 1.25rem; font-weight: 650;}
        .hr {height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 14px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load():
    model, meta = load_best_model(MODELS_DIR)
    return model, meta


def _get_score(model, meta, X: pd.DataFrame) -> np.ndarray:
    if meta.score_type == "proba":
        if not hasattr(model, "predict_proba"):
            raise ValueError("score_type='proba' nhưng model không có predict_proba().")
        return model.predict_proba(X)[:, 1]
    elif meta.score_type == "decision":
        if not hasattr(model, "decision_function"):
            raise ValueError("score_type='decision' nhưng model không có decision_function().")
        return model.decision_function(X)
    else:
        raise ValueError(f"Unknown score_type: {meta.score_type}")


def _top20_importance(model, feature_names: list[str], top_n: int = 20):
    est = model
    if hasattr(est, "named_steps"):
        est = list(est.named_steps.values())[-1]

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top_n)

    if hasattr(est, "get_booster"):
        booster = est.get_booster()
        score = booster.get_score(importance_type="gain")
        imp = np.zeros(len(feature_names), dtype=float)
        for k, v in score.items():
            if k.startswith("f"):
                idx = int(k[1:])
                if 0 <= idx < len(imp):
                    imp[idx] = float(v)
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top_n)

    return None


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(
        page_title="UNSW-NB15 IDS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Header
    st.markdown("## 🛡️ UNSW-NB15 IDS")
    st.markdown('<div class="muted">Upload CSV → preprocess → predict → (optional) metrics & feature importance</div>', unsafe_allow_html=True)
    st.write("")

    # Load model
    try:
        model, meta = _load()
    except Exception as e:
        st.error(f"Không load được best_model trong: {MODELS_DIR}\n\n{e}")
        st.stop()

    # Sidebar (gọn)
    with st.sidebar:
        st.markdown("### Model")
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Best model</div>
              <div class="card-value">{meta.best_model}</div>
              <div class="hr"></div>
              <div class="card-title">Default threshold</div>
              <div class="card-value">{float(meta.threshold):.6f}</div>
              <div class="hr"></div>
              <div class="card-title">Score type</div>
              <div class="card-value">{meta.score_type}</div>
              <div class="hr"></div>
              <div class="card-title">#features</div>
              <div class="card-value">{len(meta.features)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        thr = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(meta.threshold),
            step=0.001,
            help="Tăng threshold → giảm báo động sai (FP) nhưng có thể tăng lọt attack (FN).",
        )

        auto_preprocess = st.toggle(
            "Auto preprocess (raw UNSW CSV)",
            value=True,
            help="Bật nếu file có proto/service/state dạng text. Tắt nếu file đã one-hot sẵn đúng format.",
        )

        show_importance = st.toggle("Show Top-20 Importance", value=True)

    # Upload
    uploaded = st.file_uploader("Upload file CSV (ví dụ: UNSW_NB15_testing-set.csv)", type=["csv"])
    if not uploaded:
        st.info("Chọn file CSV để bắt đầu.")
        return

    df_raw = pd.read_csv(uploaded)

    # Preprocess
    prepared = prepare_for_best_model(
        df_raw=df_raw,
        feature_ref=meta.features,
        auto_preprocess=auto_preprocess,
    )

    # Predict
    try:
        score = _get_score(model, meta, prepared.X)
    except Exception as e:
        st.error(f"Lỗi khi tính score: {e}")
        st.stop()

    pred = (score >= float(thr)).astype(int)

    out = df_raw.copy()
    out["score"] = score
    out["pred"] = pred
    out["pred_label"] = np.where(pred == 1, "attack", "normal")

    # Top quick KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df_raw.shape[0]:,}")
    c2.metric("Raw cols", f"{df_raw.shape[1]:,}")
    c3.metric("Pred attack", f"{int((pred==1).sum()):,}")
    c4.metric("Pred normal", f"{int((pred==0).sum()):,}")

    st.write("")
    tab_overview, tab_pred, tab_metrics, tab_imp = st.tabs(["Overview", "Predictions", "Metrics", "Feature Importance"])

    # ---------------- Overview ----------------
    with tab_overview:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Data preview")
            show_preview = st.checkbox("Show preview table", value=True)
            if show_preview:
                st.dataframe(df_raw.head(30), use_container_width=True)
        with right:
            st.markdown("### Preprocess summary")
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">Model features used</div>
                  <div class="card-value">{len(meta.features)}</div>
                  <div class="hr"></div>
                  <div class="card-title">Dropped columns</div>
                  <div class="card-value">{len(prepared.dropped_cols) if prepared.dropped_cols else 0}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if prepared.dropped_cols:
                with st.expander("See dropped columns"):
                    st.write(prepared.dropped_cols)

    # ---------------- Predictions ----------------
    with tab_pred:
        st.markdown("### Predictions")
        # distribution chart
        dist = pd.Series(pred).map({0: "normal", 1: "attack"}).value_counts().reindex(["normal","attack"]).fillna(0)
        fig = plt.figure(figsize=(6.5, 3.8))
        plt.bar(dist.index, dist.values)
        plt.title("Prediction distribution")
        plt.ylabel("Count")
        plt.grid(True, axis="y", alpha=0.25)
        st.pyplot(fig)

        show_table = st.checkbox("Show output table (first 200 rows)", value=False)
        if show_table:
            st.dataframe(out.head(200), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download predictions CSV",
            data=csv_bytes,
            file_name="unsw_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ---------------- Metrics ----------------
    with tab_metrics:
        st.markdown("### Metrics (chỉ hiện khi file có label)")
        if prepared.y is None:
            st.info("File bạn upload không có cột label → không tính metrics.")
        else:
            m = ids_metrics(prepared.y.values, pred)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("DR (Recall Attack)", f"{m['DR(Recall_Attack)']:.4f}")
            mc2.metric("FPR", f"{m['FPR']:.4f}")
            mc3.metric("FAR", f"{m['FAR']:.4f}")
            mc4.metric("F1 (Attack)", f"{m['F1']:.4f}" if "F1" in m else f"{m.get('F1(Attack)', np.nan):.4f}")
            mc5.metric("AUC", f"{m['AUC']:.4f}" if "AUC" in m else "N/A")

            st.write("")

            cm = m["confusion_matrix"]
            left, right = st.columns([1, 1])
            with left:
                fig = plt.figure(figsize=(5.8, 4.6))
                sns.heatmap(
                    cm, annot=True, fmt="d", cbar=False,
                    xticklabels=["Normal(0)", "Attack(1)"],
                    yticklabels=["Normal(0)", "Attack(1)"]
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                plt.tight_layout()
                st.pyplot(fig)

            with right:
                with st.expander("Show classification report"):
                    st.text(m["classification_report"])

    # ---------------- Feature importance ----------------
    with tab_imp:
        st.markdown("### Top-20 Feature Importance")
        if not show_importance:
            st.info("Bạn đã tắt Show Top-20 Importance ở sidebar.")
        else:
            df_imp = _top20_importance(model, meta.features, top_n=20)
            if df_imp is None or df_imp["importance"].sum() == 0:
                st.warning("Model không hỗ trợ feature_importances_ (hoặc importance = 0).")
            else:
                plot_df = df_imp.sort_values("importance", ascending=True)
                fig = plt.figure(figsize=(9, 6))
                plt.barh(plot_df["feature"], plot_df["importance"])
                plt.title(f"{meta.best_model} - Top 20 Feature Importances")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.grid(True, axis="x", alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

                with st.expander("Show importance table"):
                    st.dataframe(df_imp, use_container_width=True)


if __name__ == "__main__":
    main()
