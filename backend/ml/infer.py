import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib


LABEL_BY_ID = {
    0: "not_greenwashing",
    1: "greenwashing",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_path() -> Path:
    configured = os.environ.get("ECOTEXT_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    return project_root() / "Model_Experimentation" / "artifacts" / "ecotext_tfidf_logreg.joblib"


def normalize_label(value) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"greenwashing", "not_greenwashing"}:
            return normalized
        if normalized in {"1", "true"}:
            return "greenwashing"
        if normalized in {"0", "false"}:
            return "not_greenwashing"

    try:
        return LABEL_BY_ID[int(value)]
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"Unsupported model label: {value!r}")


def read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text

    raw = sys.stdin.read()
    if not raw.strip():
        return ""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return str(payload)


def predict(text: str) -> dict:
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(path)

    prediction = model.predict([text])[0]
    label = normalize_label(prediction)

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = getattr(model, "classes_", list(range(len(proba))))
        for class_value, score in zip(classes, proba):
            probabilities[normalize_label(class_value)] = float(score)

    if not probabilities:
        probabilities = {
            "greenwashing": 1.0 if label == "greenwashing" else 0.0,
            "not_greenwashing": 1.0 if label == "not_greenwashing" else 0.0,
        }

    probabilities.setdefault("greenwashing", 0.0)
    probabilities.setdefault("not_greenwashing", 0.0)
    confidence = float(probabilities[label])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "greenwashing": float(probabilities["greenwashing"]),
            "not_greenwashing": float(probabilities["not_greenwashing"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EcoText model inference.")
    parser.add_argument("--text", help="Text to classify. If omitted, JSON or raw text is read from stdin.")
    args = parser.parse_args()

    text = read_text(args).strip()
    if not text:
        print(json.dumps({"error": "Text is required"}))
        return 2

    try:
        print(json.dumps(predict(text), ensure_ascii=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
