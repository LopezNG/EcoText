const cors = require("cors");
const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const Sentiment = require("sentiment");

const app = express();
const sentiment = new Sentiment();

const PORT = Number(process.env.PORT || process.env.API_PORT || 4000);
const PYTHON_BIN = process.env.PYTHON_BIN || process.env.PYTHON || "python";
const INFER_SCRIPT = path.join(__dirname, "ml", "infer.py");
const MAX_TEXT_LENGTH = Number(process.env.MAX_TEXT_LENGTH || 10000);

const suspiciousPhrases = [
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
];

app.use(cors({ origin: process.env.CORS_ORIGIN || true }));
app.use(express.json({ limit: "128kb" }));

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function detectSuspiciousPhrases(text) {
  const lowerText = text.toLowerCase();
  return suspiciousPhrases.filter((phrase) => lowerText.includes(phrase.toLowerCase()));
}

function highlightText(text, phrases) {
  let escaped = escapeHtml(text);
  const sortedPhrases = [...phrases].sort((a, b) => b.length - a.length);

  for (const phrase of sortedPhrases) {
    const escapedPhrase = escapeHtml(phrase);
    const pattern = new RegExp(`\\b(${escapeRegExp(escapedPhrase)})\\b`, "gi");
    escaped = escaped.replace(pattern, "<mark>$1</mark>");
  }

  return escaped;
}

function analyzeSentiment(text) {
  const result = sentiment.analyze(text);
  const normalizedScore = Math.max(-1, Math.min(1, result.comparative || 0));
  let label = "neutral";

  if (normalizedScore > 0.05) {
    label = "positive";
  } else if (normalizedScore < -0.05) {
    label = "negative";
  }

  return {
    label,
    score: Number(Math.abs(normalizedScore).toFixed(2)),
  };
}

function runModel(text) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [INFER_SCRIPT], {
      cwd: path.resolve(__dirname, ".."),
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      let payload;
      try {
        payload = JSON.parse(stdout);
      } catch (error) {
        reject(new Error(`Invalid model response. ${stderr || stdout || error.message}`));
        return;
      }

      if (code !== 0 || payload.error) {
        reject(new Error(payload.error || stderr || `Model process exited with code ${code}`));
        return;
      }

      resolve(payload);
    });

    child.stdin.write(JSON.stringify({ text }));
    child.stdin.end();
  });
}

app.get("/api/health", (_request, response) => {
  response.json({ ok: true, modelBridge: "python-child-process" });
});

app.post("/api/analyze", async (request, response) => {
  const text = typeof request.body?.text === "string" ? request.body.text.trim() : "";

  if (!text) {
    response.status(400).json({ error: "Text is required." });
    return;
  }

  if (text.length > MAX_TEXT_LENGTH) {
    response.status(413).json({ error: `Text must be ${MAX_TEXT_LENGTH} characters or fewer.` });
    return;
  }

  try {
    const modelResult = await runModel(text);
    const matchedPhrases = detectSuspiciousPhrases(text);

    response.json({
      label: modelResult.label,
      confidence: Number(modelResult.confidence.toFixed(4)),
      probabilities: {
        greenwashing: Number(modelResult.probabilities.greenwashing.toFixed(4)),
        not_greenwashing: Number(modelResult.probabilities.not_greenwashing.toFixed(4)),
      },
      sentiment: analyzeSentiment(text),
      suspicious_phrases: matchedPhrases,
      highlighted_text: highlightText(text, matchedPhrases),
    });
  } catch (error) {
    console.error("Analyze failed:", error);
    response.status(500).json({ error: "Unable to analyze text right now." });
  }
});

app.listen(PORT, () => {
  console.log(`EcoText API listening on http://localhost:${PORT}`);
});
