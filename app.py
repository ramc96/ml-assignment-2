import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Classification Model Evaluator")
st.markdown("**BITS Pilani | M.Tech AIML/DSE | Machine Learning – Assignment 2**")
st.markdown("---")

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("### Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File", type=["csv"],
    help="Upload your dataset CSV file (Heart Disease dataset recommended)"
)

st.sidebar.markdown("### Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    [
        "All Models (Comparison)",
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbor",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

st.sidebar.markdown("### Parameters")
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

# ─────────────────────────────────────────────
#  Helper: load & preprocess
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(df, target_col, test_sz, rs):
    """Encode categoricals, scale numerics, split."""
    df = df.copy().dropna()

    # encode any remaining object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # binary-encode target if needed
    if y.nunique() > 2:
        pass  # multiclass – keep as is
    else:
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=rs, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns)


def build_models(rs):
    return {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=rs),
        "Decision Tree":        DecisionTreeClassifier(random_state=rs),
        "K-Nearest Neighbor":   KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":          GaussianNB(),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=rs),
        "XGBoost":              XGBClassifier(use_label_encoder=False,
                                              eval_metric='logloss',
                                              random_state=rs, verbosity=0)
    }


def evaluate(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    multiclass = len(np.unique(y_te)) > 2
    avg = 'weighted' if multiclass else 'binary'

    try:
        if multiclass:
            y_prob = model.predict_proba(X_te)
            auc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted')
        else:
            y_prob = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = None

    return {
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "AUC":       round(auc, 4) if auc is not None else "N/A",
        "Precision": round(precision_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_te, y_pred), 4),
    }, y_pred


def plot_confusion_matrix(y_te, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload a CSV file from the sidebar to get started.")
    st.markdown("""
    ### 📋 Expected Dataset Format
    - **Recommended:** Heart Disease dataset (Kaggle) – 918 rows × 12 features
    - Target column should be the **last column** or you can select it below
    - All features should be numeric (or will be auto-encoded)

    ### 🚀 How to Use
    1. Upload your test CSV via the sidebar
    2. Select the target column
    3. Choose a model (or compare all)
    4. View metrics and confusion matrix
    """)
    st.stop()

# ── Load data ──
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("📊 Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())
st.dataframe(df.head(10), use_container_width=True)

# ── Target column selector ──
target_col = st.selectbox(
    "🎯 Select Target Column",
    options=list(df.columns),
    index=len(df.columns) - 1
)

if st.button("🚀 Run Model(s)", type="primary"):
    with st.spinner("Training and evaluating..."):
        try:
            X_tr, X_te, y_tr, y_te, feature_names = load_and_preprocess(
                df, target_col, test_size, int(random_state)
            )
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        models = build_models(int(random_state))

        if model_choice == "All Models (Comparison)":
            st.markdown("---")
            st.subheader("📈 Model Comparison")

            results = {}
            preds   = {}
            for name, mdl in models.items():
                metrics, ypred = evaluate(mdl, X_tr, X_te, y_tr, y_te)
                results[name] = metrics
                preds[name]   = ypred

            # Metrics table
            results_df = pd.DataFrame(results).T.reset_index()
            results_df.columns = ["Model"] + list(results_df.columns[1:])
            st.dataframe(results_df.style.highlight_max(
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                color="lightgreen", axis=0
            ), use_container_width=True)

            # Bar chart comparison
            st.markdown("### 📊 Visual Comparison")
            plot_df = results_df.set_index("Model")[["Accuracy", "F1 Score", "MCC"]]
            # convert to numeric safely
            plot_df = plot_df.apply(pd.to_numeric, errors='coerce')
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black')
            ax.set_title("Model Performance Comparison", fontsize=14)
            ax.set_ylabel("Score")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            st.pyplot(fig)

            # Confusion matrices
            st.markdown("### 🔲 Confusion Matrices")
            cols = st.columns(3)
            for i, (name, ypred) in enumerate(preds.items()):
                with cols[i % 3]:
                    fig_cm = plot_confusion_matrix(y_te, ypred, name)
                    st.pyplot(fig_cm)

        else:
            # Single model
            mdl  = models[model_choice]
            metrics, ypred = evaluate(mdl, X_tr, X_te, y_tr, y_te)

            st.markdown("---")
            st.subheader(f"📈 Results – {model_choice}")

            # Metric cards
            c1, c2, c3 = st.columns(3)
            c4, c5, c6 = st.columns(3)
            c1.metric("Accuracy",  metrics["Accuracy"])
            c2.metric("AUC Score", metrics["AUC"])
            c3.metric("Precision", metrics["Precision"])
            c4.metric("Recall",    metrics["Recall"])
            c5.metric("F1 Score",  metrics["F1 Score"])
            c6.metric("MCC",       metrics["MCC"])

            st.markdown("### 🔲 Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_te, ypred, model_choice)
            st.pyplot(fig_cm)

            st.markdown("### 📋 Classification Report")
            report = classification_report(y_te, ypred, output_dict=False)
            st.text(report)

            # Feature importance (tree-based)
            if model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
                st.markdown("### 🌟 Feature Importance")
                imp = mdl.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": imp
                }).sort_values("Importance", ascending=False)
                fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
                sns.barplot(data=fi_df, x="Importance", y="Feature",
                            palette="viridis", ax=ax_fi)
                ax_fi.set_title(f"Feature Importances – {model_choice}")
                plt.tight_layout()
                st.pyplot(fig_fi)

st.markdown("---")
st.markdown(
    "<center>Made for BITS Pilani ML Assignment 2 | "
    "Heart Disease Dataset | 6 Classification Models</center>",
    unsafe_allow_html=True
)
