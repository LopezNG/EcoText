import json
import math
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ROOT = Path(".")
DATA_PATH = ROOT / "emerald_data.csv"
REPORT_PATH = ROOT / "201820Sustainability20English_pdf_extracted.txt"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

LABEL_MAPPING = {"not_greenwashing": 0, "greenwashing": 1}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

SUSPICIOUS_PHRASES = [
    "sustainable solutions",
    "positive impact",
    "meaningful change",
    "better tomorrow",
    "eco-friendly",
    "green",
    "carbon neutral",
    "net zero",
    "environmentally friendly",
    "committed to sustainability",
    "reducing our footprint",
    "responsible choice",
    "sustainably",
    "climate positive",
    "nature positive",
    "planet friendly",
    "future generations",
]

POSITIVE_WORDS = {
    "accelerate",
    "achieve",
    "benefit",
    "better",
    "clean",
    "efficient",
    "good",
    "healthy",
    "improve",
    "innovation",
    "meaningful",
    "positive",
    "progress",
    "responsible",
    "safe",
    "sustainable",
    "value",
}
NEGATIVE_WORDS = {
    "bad",
    "challenge",
    "concern",
    "damage",
    "emission",
    "failure",
    "harm",
    "impact",
    "pollution",
    "problem",
    "risk",
    "scarcity",
    "waste",
}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding_artifacts(text):
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Â": "",
        "\ufeff": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def load_data():
    df = pd.read_csv(DATA_PATH)
    required = {"claim", "gold_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["claim_clean"] = df["claim"].map(clean_text)
    df = df[df["claim_clean"].ne("")]
    df["label"] = df["gold_label"].map(LABEL_MAPPING)
    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "gold_label"].dropna().unique())
        raise ValueError(f"Unexpected labels: {bad}")
    df["label"] = df["label"].astype(int)
    return df


def split_data(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_positive_scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        return 1 / (1 + np.exp(-scores))
    return None


def evaluate_model(name, model, texts, y_true, split_name):
    y_pred = model.predict(texts)
    pos_scores = get_positive_scores(model, texts)
    row = {
        "model": name,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_greenwashing": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_greenwashing": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_greenwashing": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "tn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, y_pred, labels=[0, 1])[1, 1]),
        "roc_auc": np.nan,
        "average_precision_pr_auc": np.nan,
        "brier_score": np.nan,
    }
    if pos_scores is not None and len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, pos_scores)
        row["average_precision_pr_auc"] = average_precision_score(y_true, pos_scores)
        row["brier_score"] = brier_score_loss(y_true, pos_scores)
    return row


def build_models():
    base_tfidf = {
        "tfidf__ngram_range": (1, 2),
        "tfidf__min_df": 2,
        "tfidf__max_df": 0.95,
        "tfidf__sublinear_tf": True,
    }
    models = {
        "TF-IDF + Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ).set_params(**base_tfidf),
        "TF-IDF + Calibrated Linear SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", random_state=SEED),
                        cv=3,
                    ),
                ),
            ]
        ).set_params(**base_tfidf),
        "TF-IDF + Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True)),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ).set_params(**base_tfidf),
        "TF-IDF + Random Forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, max_features=5000)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["TF-IDF + XGBoost"] = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, max_features=6000)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    return models


def simple_sentiment(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    if not tokens:
        return "neutral", 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / math.sqrt(len(tokens))
    if score >= 0.15:
        label = "positive"
    elif score <= -0.15:
        label = "negative"
    else:
        label = "neutral"
    return label, float(score)


def detect_suspicious_phrases(text, phrases=SUSPICIOUS_PHRASES):
    found = []
    lower = str(text).lower()
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
        if re.search(pattern, lower):
            found.append(phrase)
    return found


def highlight_suspicious_phrases(text, phrases=SUSPICIOUS_PHRASES):
    highlighted = str(text)
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(r"(?i)(?<!\w)(" + re.escape(phrase) + r")(?!\w)")
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)
    return highlighted


def chunk_report(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = fix_encoding_artifacts(raw)
    lines = [clean_text(line) for line in raw.splitlines()]
    drop_patterns = [
        r"^ecolab sustainability report 2018\s+\d+$",
        r"^accelerating meaningful change contents$",
    ]
    kept = []
    for line in lines:
        if not line:
            kept.append("")
            continue
        if any(re.match(pattern, line, flags=re.I) for pattern in drop_patterns):
            continue
        kept.append(line)
    paragraphs = []
    current = []
    for line in kept:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    chunks = []
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if len(paragraph) < 40:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        buffer = []
        for sentence in sentences:
            sentence = clean_text(sentence)
            if not sentence:
                continue
            buffer.append(sentence)
            joined = " ".join(buffer)
            if len(joined) >= 180:
                chunks.append(joined)
                buffer = []
        if buffer:
            chunks.append(" ".join(buffer))
    return [chunk for chunk in chunks if len(chunk) >= 50]


def train_and_export():
    df = load_data()
    train_df, val_df, test_df = split_data(df)
    models = build_models()
    rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(train_df["claim_clean"], train_df["label"])
        fitted_models[name] = model
        rows.append(evaluate_model(name, model, val_df["claim_clean"], val_df["label"], "validation"))
        rows.append(evaluate_model(name, model, test_df["claim_clean"], test_df["label"], "test"))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ARTIFACT_DIR / "model_comparison_metrics.csv", index=False)

    val_rank = metrics_df[metrics_df["split"].eq("validation")].sort_values(
        ["f1_greenwashing", "macro_f1", "average_precision_pr_auc"],
        ascending=False,
        na_position="last",
    )
    best_name = val_rank.iloc[0]["model"]
    deployable_name = "TF-IDF + Logistic Regression"

    final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    selected_model = clone(fitted_models[deployable_name])
    selected_model.fit(final_train_df["claim_clean"], final_train_df["label"])
    joblib.dump(selected_model, ARTIFACT_DIR / "ecotext_tfidf_logreg.joblib")
    (ARTIFACT_DIR / "label_mapping.json").write_text(json.dumps(LABEL_MAPPING, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "suspicious_phrases.json").write_text(json.dumps(SUSPICIOUS_PHRASES, indent=2), encoding="utf-8")

    chunks = chunk_report(REPORT_PATH)
    chunk_df = pd.DataFrame({"chunk_text": chunks})
    if not chunk_df.empty:
        proba = selected_model.predict_proba(chunk_df["chunk_text"])[:, 1]
        pred = (proba >= 0.5).astype(int)
        sentiments = chunk_df["chunk_text"].map(simple_sentiment)
        chunk_df["predicted_label"] = [INVERSE_LABEL_MAPPING[int(label)] for label in pred]
        chunk_df["confidence"] = np.where(pred == 1, proba, 1 - proba)
        chunk_df["greenwashing_probability"] = proba
        chunk_df["sentiment_label"] = [item[0] for item in sentiments]
        chunk_df["sentiment_score"] = [item[1] for item in sentiments]
        chunk_df["suspicious_phrases"] = chunk_df["chunk_text"].map(lambda text: "; ".join(detect_suspicious_phrases(text)))
        chunk_df["highlighted_html"] = chunk_df["chunk_text"].map(highlight_suspicious_phrases)
        top_chunks = chunk_df.sort_values(
            ["greenwashing_probability", "suspicious_phrases"],
            ascending=[False, False],
        ).head(50)
    else:
        top_chunks = chunk_df
    top_chunks.to_csv(ARTIFACT_DIR / "top_suspicious_chunks.csv", index=False)

    notes = f"""# EcoText final notes

Recommended first web-app model: TF-IDF + Logistic Regression.

Why: it is fast, simple to deploy with joblib, provides probability-style confidence scores, and is easier to explain for a student/final-year project demonstration. The notebook also includes calibrated Linear SVM, Naive Bayes, Random Forest, optional XGBoost, optional Sentence-BERT, and optional transformer fine-tuning sections for comparison.

Best validation model in this run: {best_name}.

Use sentiment only as an auxiliary signal. Positive sustainability language can appear in both real claims and greenwashing, so sentiment must not override the classifier.
"""
    (ARTIFACT_DIR / "final_notes.md").write_text(notes, encoding="utf-8")
    return df, train_df, val_df, test_df, metrics_df


def nb_cell(cell_type, source):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.strip("\n").splitlines(True)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def make_notebook():
    markdown_intro = """# EcoText - Sustainability Text Analyzer

This notebook builds a reproducible NLP workflow for classifying sustainability-related claims as `greenwashing` or `not_greenwashing`.

The supervised training data is `emerald_data.csv`. The sustainability report TXT file is treated as unlabeled raw text only: it is used for chunking, inference demos, sentiment scoring, suspicious phrase highlighting, and manual-review exports.
"""

    setup_code = r"""import json
import math
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("emerald_data.csv")
REPORT_PATH = Path("201820Sustainability20English_pdf_extracted.txt")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

LABEL_MAPPING = {"not_greenwashing": 0, "greenwashing": 1}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

SUSPICIOUS_PHRASES = [
    "sustainable solutions", "positive impact", "meaningful change",
    "better tomorrow", "eco-friendly", "green", "carbon neutral",
    "net zero", "environmentally friendly", "committed to sustainability",
    "reducing our footprint", "responsible choice", "sustainably",
    "climate positive", "nature positive", "planet friendly",
    "future generations"
]"""

    helper_code = r"""def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding_artifacts(text):
    replacements = {
        "â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€": '"',
        "â€“": "-", "â€”": "-", "Â": "", "\ufeff": ""
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def detect_suspicious_phrases(text, phrases=SUSPICIOUS_PHRASES):
    found = []
    lower = str(text).lower()
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
        if re.search(pattern, lower):
            found.append(phrase)
    return found


def highlight_suspicious_phrases(text, phrases=SUSPICIOUS_PHRASES):
    highlighted = str(text)
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(r"(?i)(?<!\w)(" + re.escape(phrase) + r")(?!\w)")
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)
    return highlighted


def get_positive_scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        return 1 / (1 + np.exp(-scores))
    return None


def evaluate_model(name, model, texts, y_true, split_name):
    y_pred = model.predict(texts)
    pos_scores = get_positive_scores(model, texts)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row = {
        "model": name,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_greenwashing": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_greenwashing": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_greenwashing": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "roc_auc": np.nan,
        "average_precision_pr_auc": np.nan,
        "brier_score": np.nan,
    }
    if pos_scores is not None and len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, pos_scores)
        row["average_precision_pr_auc"] = average_precision_score(y_true, pos_scores)
        row["brier_score"] = brier_score_loss(y_true, pos_scores)
    return row


def show_errors(model, frame, n=5):
    temp = frame[["claim_clean", "gold_label", "label"]].copy()
    temp["pred"] = model.predict(temp["claim_clean"])
    temp["greenwashing_probability"] = get_positive_scores(model, temp["claim_clean"])
    fp = temp[(temp["label"] == 0) & (temp["pred"] == 1)].sort_values("greenwashing_probability", ascending=False)
    fn = temp[(temp["label"] == 1) & (temp["pred"] == 0)].sort_values("greenwashing_probability", ascending=True)
    print("False positives: predicted greenwashing but actually not_greenwashing")
    display(fp.head(n))
    print("False negatives: predicted not_greenwashing but actually greenwashing")
    display(fn.head(n))"""

    data_code = r"""df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())

print("Columns:", df.columns.tolist())
display(df.isna().sum().to_frame("missing_values"))
print("Duplicate claims:", df["claim"].duplicated().sum())
display(df["gold_label"].value_counts().to_frame("count"))

df["claim_clean"] = df["claim"].map(clean_text)
df["label"] = df["gold_label"].map(LABEL_MAPPING).astype(int)

print("Samples from each class:")
display(df.groupby("gold_label", group_keys=False).sample(n=3, random_state=SEED)[["claim", "gold_label", "justification"]])

# Heuristic check only: these are not automatic errors, just rows worth reading.
specific_words = r"\b(approximately|\d|percent|%|tonnes?|million|billion|reduced|increased|measured|certified)\b"
vague_words = r"\b(green|eco-friendly|better tomorrow|positive impact|meaningful change|sustainable solutions|committed)\b"
mismatch_candidates = df[
    ((df["gold_label"].eq("greenwashing")) & df["claim_clean"].str.contains(specific_words, case=False, regex=True, na=False)) |
    ((df["gold_label"].eq("not_greenwashing")) & df["claim_clean"].str.contains(vague_words, case=False, regex=True, na=False))
]
print("Heuristic label/claim review candidates:", len(mismatch_candidates))
display(mismatch_candidates[["claim", "gold_label", "justification"]].head(15))"""

    split_code = r"""train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

for name, frame in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(name, frame.shape)
    display(frame["gold_label"].value_counts(normalize=True).rename("proportion").to_frame())"""

    models_code = r"""base_tfidf = {
    "tfidf__ngram_range": (1, 2),
    "tfidf__min_df": 2,
    "tfidf__max_df": 0.95,
    "tfidf__sublinear_tf": True,
}

models = {
    "TF-IDF + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]).set_params(**base_tfidf),
    "TF-IDF + Calibrated Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3)),
    ]).set_params(**base_tfidf),
    "TF-IDF + Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", MultinomialNB(alpha=0.5)),
    ]).set_params(**base_tfidf),
    "TF-IDF + Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]),
}

try:
    from xgboost import XGBClassifier
    models["TF-IDF + XGBoost"] = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, max_features=6000)),
        ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=SEED)),
    ])
except Exception as exc:
    print("Skipping XGBoost:", exc)

rows = []
fitted_models = {}
for name, model in models.items():
    print(f"Training {name}")
    model.fit(train_df["claim_clean"], train_df["label"])
    fitted_models[name] = model
    rows.append(evaluate_model(name, model, val_df["claim_clean"], val_df["label"], "validation"))
    rows.append(evaluate_model(name, model, test_df["claim_clean"], test_df["label"], "test"))

metrics_df = pd.DataFrame(rows)
display(metrics_df.sort_values(["split", "f1_greenwashing", "macro_f1"], ascending=[True, False, False]))
metrics_df.to_csv(ARTIFACT_DIR / "model_comparison_metrics.csv", index=False)"""

    confidence_code = r"""val_metrics = metrics_df[metrics_df["split"].eq("validation")].sort_values(
    ["f1_greenwashing", "macro_f1", "average_precision_pr_auc"],
    ascending=False,
    na_position="last",
)
best_validation_name = val_metrics.iloc[0]["model"]
best_validation_model = fitted_models[best_validation_name]
print("Best validation model:", best_validation_name)

deployable_name = "TF-IDF + Logistic Regression"
deployable_model = fitted_models[deployable_name]

for name in [deployable_name, best_validation_name]:
    model = fitted_models[name]
    y_score = get_positive_scores(model, test_df["claim_clean"])
    if y_score is None:
        continue
    prob_true, prob_pred = calibration_curve(test_df["label"], y_score, n_bins=5)
    plt.plot(prob_pred, prob_true, marker="o", label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve")
plt.legend()
plt.show()

print("Classification report for deployable model:")
print(classification_report(test_df["label"], deployable_model.predict(test_df["claim_clean"]), target_names=["not_greenwashing", "greenwashing"]))
show_errors(deployable_model, test_df, n=5)"""

    sbert_code = r"""try:
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_emb = embedder.encode(train_df["claim_clean"].tolist(), show_progress_bar=True)
    X_val_emb = embedder.encode(val_df["claim_clean"].tolist(), show_progress_bar=True)
    X_test_emb = embedder.encode(test_df["claim_clean"].tolist(), show_progress_bar=True)

    sbert_lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    sbert_lr.fit(X_train_emb, train_df["label"])

    sbert_rows = [
        evaluate_model("Sentence-BERT embeddings + Logistic Regression", sbert_lr, X_val_emb, val_df["label"], "validation"),
        evaluate_model("Sentence-BERT embeddings + Logistic Regression", sbert_lr, X_test_emb, test_df["label"], "test"),
    ]
    display(pd.DataFrame(sbert_rows))
except Exception as exc:
    print("Sentence-BERT section skipped.")
    print("Install with: pip install sentence-transformers")
    print("Reason:", exc)"""

    transformer_code = r"""# Optional advanced model. This section is intentionally compact because small datasets can overfit quickly.
# Install if needed: pip install transformers torch datasets accelerate evaluate
try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=192)

    train_ds = Dataset.from_pandas(train_df[["claim_clean", "label"]].rename(columns={"claim_clean": "text"}))
    val_ds = Dataset.from_pandas(val_df[["claim_clean", "label"]].rename(columns={"claim_clean": "text"}))
    test_ds = Dataset.from_pandas(test_df[["claim_clean", "label"]].rename(columns={"claim_clean": "text"}))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_greenwashing": f1_score(labels, preds, pos_label=1, zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "average_precision": average_precision_score(labels, probs),
        }

    args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "distilbert_greenwashing"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=bert_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate(test_ds))
except Exception as exc:
    print("Transformer fine-tuning section skipped.")
    print("Install with: pip install transformers torch datasets accelerate evaluate")
    print("Reason:", exc)"""

    sentiment_code = r"""POSITIVE_WORDS = {
    "accelerate", "achieve", "benefit", "better", "clean", "efficient",
    "good", "healthy", "improve", "innovation", "meaningful", "positive",
    "progress", "responsible", "safe", "sustainable", "value"
}
NEGATIVE_WORDS = {
    "bad", "challenge", "concern", "damage", "emission", "failure", "harm",
    "impact", "pollution", "problem", "risk", "scarcity", "waste"
}

def simple_sentiment(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    if not tokens:
        return "neutral", 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / math.sqrt(len(tokens))
    if score >= 0.15:
        label = "positive"
    elif score <= -0.15:
        label = "negative"
    else:
        label = "neutral"
    return label, float(score)

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

    def sentiment_analyze(text):
        compound = vader.polarity_scores(text)["compound"]
        if compound >= 0.05:
            return "positive", compound
        if compound <= -0.05:
            return "negative", compound
        return "neutral", compound
except Exception as exc:
    print("VADER unavailable; using a small transparent fallback lexicon.")
    print("To use VADER: pip install nltk, then run nltk.download('vader_lexicon')")
    sentiment_analyze = simple_sentiment

def analyze_claim(text, model=deployable_model):
    cleaned = clean_text(text)
    proba_green = model.predict_proba([cleaned])[0, 1]
    pred = int(proba_green >= 0.5)
    sent_label, sent_score = sentiment_analyze(cleaned)
    return {
        "text": cleaned,
        "predicted_label": INVERSE_LABEL_MAPPING[pred],
        "greenwashing_confidence": float(proba_green if pred == 1 else 1 - proba_green),
        "greenwashing_probability": float(proba_green),
        "sentiment_label": sent_label,
        "sentiment_score": sent_score,
        "suspicious_phrases": detect_suspicious_phrases(cleaned),
        "highlighted_html": highlight_suspicious_phrases(cleaned),
    }

demo = "We are committed to sustainability and delivering sustainable solutions for a better tomorrow."
analyze_claim(demo)"""

    report_code = r"""def chunk_report(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = fix_encoding_artifacts(raw)
    lines = [clean_text(line) for line in raw.splitlines()]
    drop_patterns = [
        r"^ecolab sustainability report 2018\s+\d+$",
        r"^accelerating meaningful change contents$",
    ]
    kept = []
    for line in lines:
        if not line:
            kept.append("")
            continue
        if any(re.match(pattern, line, flags=re.I) for pattern in drop_patterns):
            continue
        kept.append(line)

    paragraphs = []
    current = []
    for line in kept:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    chunks = []
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if len(paragraph) < 40:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        buffer = []
        for sentence in sentences:
            sentence = clean_text(sentence)
            if not sentence:
                continue
            buffer.append(sentence)
            joined = " ".join(buffer)
            if len(joined) >= 180:
                chunks.append(joined)
                buffer = []
        if buffer:
            chunks.append(" ".join(buffer))
    return [chunk for chunk in chunks if len(chunk) >= 50]

chunks = chunk_report(REPORT_PATH)
print("Chunks:", len(chunks))
chunk_df = pd.DataFrame({"chunk_text": chunks})

proba = deployable_model.predict_proba(chunk_df["chunk_text"])[:, 1]
pred = (proba >= 0.5).astype(int)
sentiments = chunk_df["chunk_text"].map(sentiment_analyze)

chunk_df["predicted_label"] = [INVERSE_LABEL_MAPPING[int(label)] for label in pred]
chunk_df["confidence"] = np.where(pred == 1, proba, 1 - proba)
chunk_df["greenwashing_probability"] = proba
chunk_df["sentiment_label"] = [item[0] for item in sentiments]
chunk_df["sentiment_score"] = [item[1] for item in sentiments]
chunk_df["suspicious_phrases"] = chunk_df["chunk_text"].map(lambda text: "; ".join(detect_suspicious_phrases(text)))
chunk_df["highlighted_html"] = chunk_df["chunk_text"].map(highlight_suspicious_phrases)

top_suspicious_chunks = chunk_df.sort_values(
    ["greenwashing_probability", "suspicious_phrases"],
    ascending=[False, False],
).head(50)
display(top_suspicious_chunks)
top_suspicious_chunks.to_csv(ARTIFACT_DIR / "top_suspicious_chunks.csv", index=False)"""

    save_code = r"""final_train_df = pd.concat([train_df, val_df], ignore_index=True)
selected_model = clone(fitted_models[deployable_name])
selected_model.fit(final_train_df["claim_clean"], final_train_df["label"])

joblib.dump(selected_model, ARTIFACT_DIR / "ecotext_tfidf_logreg.joblib")
(ARTIFACT_DIR / "label_mapping.json").write_text(json.dumps(LABEL_MAPPING, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "suspicious_phrases.json").write_text(json.dumps(SUSPICIOUS_PHRASES, indent=2), encoding="utf-8")

print("Saved:")
print(ARTIFACT_DIR / "ecotext_tfidf_logreg.joblib")
print(ARTIFACT_DIR / "label_mapping.json")
print(ARTIFACT_DIR / "suspicious_phrases.json")
print(ARTIFACT_DIR / "model_comparison_metrics.csv")
print(ARTIFACT_DIR / "top_suspicious_chunks.csv")"""

    recommendation_md = """## Recommendation

For the first deployable EcoText web app, use **TF-IDF + Logistic Regression**.

It is fast, simple to save/load with `joblib`, has usable `predict_proba` confidence scores, and is easy to explain during a final-year project demonstration. Use the DistilBERT/RoBERTa section as an advanced comparison when GPU/runtime dependencies are available.

When interpreting outputs, prioritize greenwashing F1, greenwashing recall, macro F1, and PR-AUC. Accuracy alone can hide weak greenwashing detection because the dataset is imbalanced.

Sentiment is deliberately separate from classification. Positive sentiment does **not** mean a claim is not greenwashing; greenwashing often uses positive and vague sustainability language.
"""

    cells = [
        nb_cell("markdown", markdown_intro),
        nb_cell("markdown", "## 1. Setup and Reproducibility"),
        nb_cell("code", setup_code),
        nb_cell("code", helper_code),
        nb_cell("markdown", "## 2. Data Loading and Inspection"),
        nb_cell("code", data_code),
        nb_cell("markdown", "## 3. Stratified Train/Validation/Test Split\n\nThe report TXT is not used in supervised training, which prevents leakage from unlabeled corporate report text into the classifier."),
        nb_cell("code", split_code),
        nb_cell("markdown", "## 4. Classic ML Baselines\n\nThese models use conservative preprocessing: whitespace cleanup plus TF-IDF lowercasing inside the vectorizer."),
        nb_cell("code", models_code),
        nb_cell("markdown", "## 5. Metrics, Confidence, Calibration, and Error Analysis"),
        nb_cell("code", confidence_code),
        nb_cell("markdown", "## 6. Optional Sentence-BERT Embeddings"),
        nb_cell("code", sbert_code),
        nb_cell("markdown", "## 7. Optional Transformer Fine-Tuning"),
        nb_cell("code", transformer_code),
        nb_cell("markdown", "## 8. Sentiment + Classification Combo and Suspicious Phrase Highlighting"),
        nb_cell("code", sentiment_code),
        nb_cell("markdown", "## 9. TXT Sustainability Report Processing"),
        nb_cell("code", report_code),
        nb_cell("markdown", "## 10. Save Deployable Artifacts"),
        nb_cell("code", save_code),
        nb_cell("markdown", recommendation_md),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "EcoText_Sustainability_Text_Analyzer.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    df, train_df, val_df, test_df, metrics_df = train_and_export()
    make_notebook()
    print("Created EcoText_Sustainability_Text_Analyzer.ipynb")
    print("Created artifacts in", ARTIFACT_DIR.resolve())
    print(metrics_df.sort_values(["split", "f1_greenwashing", "macro_f1"], ascending=[True, False, False]).to_string(index=False))
