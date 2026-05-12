# EcoText - Sustainability Text Analyzer

EcoText analyzes sustainability-related text and classifies it as `greenwashing` or `not_greenwashing` using the saved TF-IDF + Logistic Regression model.

## Requirements

- Node.js 18+
- Python 3.10+
- Python packages for the saved scikit-learn model:

```bash
pip install scikit-learn joblib
```

The model artifact is expected at:

```text
Model_Experimentation/artifacts/ecotext_tfidf_logreg.joblib
```

You can override that with `ECOTEXT_MODEL_PATH` if needed.

## Setup

```bash
npm install
```

## Run the App

Start the Node.js API backend:

```bash
npm run api
```

In another terminal, start the Next.js frontend:

```bash
npm run dev
```

Open the frontend at:

```text
http://localhost:3000
```

The frontend calls the API at `http://localhost:4000` by default. To use a different API URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:4000 npm run dev
```

On Windows PowerShell:

```powershell
$env:NEXT_PUBLIC_API_URL="http://localhost:4000"; npm run dev
```

## API

Health check:

```bash
curl http://localhost:4000/api/health
```

Analyze text:

```bash
curl -X POST http://localhost:4000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Our eco-friendly product is carbon neutral and committed to sustainability.\"}"
```

Response shape:

```json
{
  "label": "greenwashing",
  "confidence": 0.87,
  "probabilities": {
    "greenwashing": 0.87,
    "not_greenwashing": 0.13
  },
  "sentiment": {
    "label": "positive",
    "score": 0.72
  },
  "suspicious_phrases": ["eco-friendly"],
  "highlighted_text": "Our <mark>eco-friendly</mark> product..."
}
```

Sentiment is only auxiliary context and does not change the classifier output.

## Direct Model Bridge Test

```bash
python backend/ml/infer.py --text "Our eco-friendly product is carbon neutral."
```

## Sample Inputs

- Quantified claim: `Our 2025 packaging redesign reduced plastic use by 32% compared with 2022 levels, verified by our annual supplier audit.`
- Vague claim: `Our eco-friendly product is a responsible choice for a better tomorrow, delivering sustainable solutions and positive impact.`
- Neutral factual claim: `The company opened a new distribution center in March and hired 40 additional logistics employees.`
