"use client";

import Image from "next/image";
import {
  ArrowDownUp,
  ArrowRight,
  Award,
  Ban,
  CircleX,
  CloudFog,
  FileSearch,
  Info,
  Leaf,
  Scale,
  Search,
  ShieldAlert,
  Sun,
  Zap,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { useMemo, useState } from "react";

const examples = [
  {
    title: "Example Tweet",
    text: "\"Our new sneakers are 100% eco-friendly and carbon neutral! #GreenFashion\"",
  },
  {
    title: "Product Blurb",
    text: "\"Made with all-natural ingredients, our formula has zero environmental impact.\"",
  },
  {
    title: "Article Excerpt",
    text: "\"The company claims to offset all emissions, but no third-party audit was cited.\"",
  },
];

const defaultText =
  "Our new sneakers are 100% natural and have zero impact on the environment. We are committed to a greener future with sustainable practices.";

const sins = [
  {
    title: "The Sin of the Hidden Trade-off",
    description:
      "Claiming a product is 'green' based on a narrow set of attributes while ignoring other significant environmental issues.",
    example:
      "Paper made from sustainably harvested trees, but ignoring the pollution from its manufacturing process.",
    icon: Scale,
    tone: "green",
  },
  {
    title: "The Sin of No Proof",
    description:
      "Making an environmental claim that cannot be substantiated by easily accessible supporting information or a reliable third-party certification.",
    example:
      "A shampoo labeled 'not tested on animals' with no certification or verifiable proof to back up the claim.",
    icon: FileSearch,
    tone: "orange",
  },
  {
    title: "The Sin of Vagueness",
    description:
      "Using broad or poorly defined terms that are likely to be misunderstood by the consumer.",
    example: "A product marketed as 'all-natural' - arsenic and mercury are natural too, but toxic.",
    icon: CloudFog,
    tone: "green-light",
  },
  {
    title: "The Sin of Irrelevance",
    description:
      "Making an environmental claim that may be truthful but is unimportant or unhelpful for consumers seeking genuinely eco-friendly products.",
    example: "A product advertised as 'CFC-free' even though CFCs have been banned by law for decades.",
    icon: Ban,
    tone: "orange",
  },
  {
    title: "The Sin of Lesser of Two Evils",
    description:
      "Making a 'green' claim that may be true within the product category, but risks distracting the consumer from the greater environmental impact of the category as a whole.",
    example: "'Eco-friendly' cigarettes or 'organic' pesticides - green claims on inherently harmful products.",
    icon: ArrowDownUp,
    tone: "red",
  },
  {
    title: "The Sin of Fibbing",
    description: "Making environmental claims that are simply false. This is the least frequent sin, but it does occur.",
    example: "A product falsely claiming to be Energy Star certified when it has no such certification.",
    icon: CircleX,
    tone: "red",
  },
  {
    title: "The Sin of Worshiping False Labels",
    description:
      "Creating a false impression of third-party endorsement through fake labels, certifications, or misleading imagery that suggests official approval.",
    example:
      "A paper plate with a self-made 'eco-certified' green seal that resembles official certifications but has no backing.",
    icon: Award,
    tone: "orange",
  },
];

type Screen = "analyze" | "results" | "learn";

const offModeBackgrounds: Record<Screen, { src: string; className: string }> = {
  analyze: { src: "/background/Landing.gif", className: "landing-backdrop" },
  results: { src: "/background/Analysis_Result.gif", className: "results-backdrop" },
  learn: { src: "/background/Greenwashing101.gif", className: "learn-backdrop" },
};

export default function Home() {
  const [screen, setScreen] = useState<Screen>("analyze");
  const [text, setText] = useState("");
  const [feedback, setFeedback] = useState<"yes" | "no" | null>(null);
  const [isLowCarbon, setIsLowCarbon] = useState(true);

  const enteredText = useMemo(() => text.trim() || defaultText, [text]);
  const background = offModeBackgrounds[screen];

  function analyze() {
    setFeedback(null);
    setScreen("results");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function goTo(next: Screen) {
    setScreen(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <main className={isLowCarbon ? "app low-carbon-on" : "app low-carbon-off"}>
      {!isLowCarbon && <AnimatedBackdrop src={background.src} className={background.className} />}
      <NavBar onNavigate={goTo} isLowCarbon={isLowCarbon} onToggleMode={() => setIsLowCarbon((value) => !value)} />
      {screen === "analyze" && <AnalyzePage text={text} setText={setText} onAnalyze={analyze} />}
      {screen === "results" && (
        <ResultsPage analyzedText={enteredText} onLearn={() => goTo("learn")} feedback={feedback} setFeedback={setFeedback} />
      )}
      {screen === "learn" && <LearnPage onStart={() => goTo("analyze")} />}
    </main>
  );
}

function AnimatedBackdrop({ src, className }: { src: string; className: string }) {
  return (
    <div className={`animated-backdrop ${className}`} aria-hidden="true">
      <Image src={src} alt="" fill sizes="100vw" priority unoptimized />
    </div>
  );
}

function NavBar({
  onNavigate,
  isLowCarbon,
  onToggleMode,
}: {
  onNavigate: (screen: Screen) => void;
  isLowCarbon: boolean;
  onToggleMode: () => void;
}) {
  return (
    <header className="nav">
      <button className="logo" onClick={() => onNavigate("analyze")} aria-label="EcoText home">
        <Leaf size={24} aria-hidden="true" />
        <span>EcoText</span>
      </button>
      <nav className="nav-center" aria-label="Primary">
        <button onClick={() => onNavigate("analyze")}>Analyze</button>
        <button onClick={() => onNavigate("learn")}>Learn</button>
        <button type="button">About</button>
      </nav>
      <div className="nav-right">
        <button className="mode-toggle" type="button" aria-pressed={isLowCarbon} onClick={onToggleMode}>
          {isLowCarbon ? <Sun size={16} aria-hidden="true" /> : <Zap size={16} aria-hidden="true" />}
          <span>Low-Carbon Mode</span>
          <span className={isLowCarbon ? "switch on" : "switch off"} aria-hidden="true">
            <span />
          </span>
        </button>
        <button className="feedback" type="button">Submit Feedback</button>
      </div>
    </header>
  );
}

function AnalyzePage({
  text,
  setText,
  onAnalyze,
}: {
  text: string;
  setText: (text: string) => void;
  onAnalyze: () => void;
}) {
  return (
    <section className="page landing" aria-labelledby="hero-title">
      <div className="hero">
        <h1 id="hero-title">Scan Your Text for Sustainability.</h1>
        <p>Analyze tweets, articles, and product descriptions to verify eco-friendly claims and detect greenwashing.</p>
      </div>

      <section className="input-wrap" aria-label="Text analyzer">
        <div className="analyzer-card">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value.slice(0, 1000))}
            placeholder="Paste your text here (product description, article, tweet)..."
            maxLength={1000}
            aria-label="Text to analyze"
          />
          <div className="counter">{text.length} / 1000</div>
          <button className="primary-button" onClick={onAnalyze}>Analyze Now</button>
        </div>
      </section>

      <section className="examples" aria-labelledby="examples-title">
        <h2 id="examples-title">Try an example</h2>
        <div className="example-row">
          {examples.map((example) => (
            <button className="example-tile" key={example.title} onClick={() => setText(example.text)}>
              <strong>{example.title}</strong>
              <span>{example.text}</span>
            </button>
          ))}
        </div>
      </section>
    </section>
  );
}

function ResultsPage({
  analyzedText,
  onLearn,
  feedback,
  setFeedback,
}: {
  analyzedText: string;
  onLearn: () => void;
  feedback: "yes" | "no" | null;
  setFeedback: (value: "yes" | "no") => void;
}) {
  return (
    <section className="page results" aria-labelledby="results-title">
      <div className="results-body">
        <div className="left-col">
          <section className="verdict-card" aria-labelledby="results-title">
            <p>Classification</p>
            <h1 id="results-title">GREENWASHING</h1>
            <span>The analyzed text contains claims that may be misleading or unsubstantiated.</span>
          </section>

          <section className="card confidence" aria-labelledby="confidence-title">
            <div className="row between">
              <h2 id="confidence-title">Confidence Score</h2>
              <strong>85%</strong>
            </div>
            <div className="progress" aria-label="Confidence score 85 percent">
              <span />
            </div>
          </section>

          <section className="card heatmap" aria-labelledby="heatmap-title">
            <div className="row heat-title">
              <h2 id="heatmap-title">Explainability Heatmap (LIME/SHAP Analysis)</h2>
              <Info size={16} aria-hidden="true" />
            </div>
            <div className="heat-body">
              <p>
                {splitAnalyzedText(analyzedText).before}
                <mark className="orange">100% natural</mark>
                {splitAnalyzedText(analyzedText).middle}
                <mark className="red">zero impact</mark>
                {splitAnalyzedText(analyzedText).after}
              </p>
            </div>
            <div className="legend">
              <span><i className="dot orange-dot" />Vague Claim</span>
              <span><i className="dot red-dot" />Unsubstantiated</span>
            </div>
          </section>

          <section className="card crowd" aria-labelledby="crowd-title">
            <h2 id="crowd-title">Was this classification accurate?</h2>
            <div className="crowd-buttons">
              <button className={feedback === "yes" ? "selected good" : "good"} onClick={() => setFeedback("yes")}>
                <ThumbsUp size={18} aria-hidden="true" /> Yes, accurate
              </button>
              <button className={feedback === "no" ? "selected bad" : "bad"} onClick={() => setFeedback("no")}>
                <ThumbsDown size={18} aria-hidden="true" /> No, incorrect
              </button>
            </div>
          </section>
        </div>

        <aside className="right-col" aria-label="Analysis details">
          <section className="card tactic">
            <p>Potential Greenwashing Tactic Detected</p>
            <strong>The Sin of No Proof</strong>
            <span>An environmental claim that cannot be substantiated by easily accessible supporting information or by a reliable third-party certification.</span>
            <button onClick={onLearn}>Learn more on Greenwashing 101 <ArrowRight size={14} aria-hidden="true" /></button>
          </section>

          <section className="card company">
            <h2>Optional Company Context</h2>
            <p><strong>Company:</strong> GreenShoe Co.</p>
            <p><strong>ESG Risk:</strong> <span>Moderate</span></p>
            <small>This is a prototype value. Real ESG data would be sourced from third-party providers.</small>
          </section>
        </aside>
      </div>
    </section>
  );
}

function splitAnalyzedText(text: string) {
  const fallback = {
    before: "Our new sneakers are ",
    middle: " and have ",
    after: " on the environment. We are committed to a greener future with sustainable practices.",
  };

  if (!text.includes("100% natural") || !text.includes("zero impact")) {
    return fallback;
  }

  const [before, rest] = text.split("100% natural");
  const [middle, after] = rest.split("zero impact");
  return { before, middle, after };
}

function LearnPage({ onStart }: { onStart: () => void }) {
  return (
    <section className="page learn" aria-labelledby="learn-title">
      <div className="learn-content">
        <header className="learn-header">
          <ShieldAlert size={48} aria-hidden="true" />
          <h1 id="learn-title">Understanding the Seven Sins of Greenwashing</h1>
          <p>Learn to identify common tactics used to mislead consumers about environmental practices.</p>
        </header>
        <div className="divider" />
        <div className="sins-grid">
          {sins.map((sin) => (
            <SinCard key={sin.title} {...sin} />
          ))}
        </div>
        <section className="footer-cta" aria-labelledby="footer-cta-title">
          <Search size={32} aria-hidden="true" />
          <h2 id="footer-cta-title">Ready to analyze your text?</h2>
          <p>Use our analyzer to detect greenwashing in any product description, article, or social media post.</p>
          <button className="primary-button with-icon" onClick={onStart}>
            Start Analyzing <ArrowRight size={16} aria-hidden="true" />
          </button>
        </section>
      </div>
    </section>
  );
}

function SinCard({
  title,
  description,
  example,
  icon: Icon,
  tone,
}: {
  title: string;
  description: string;
  example: string;
  icon: typeof Scale;
  tone: string;
}) {
  return (
    <article className="sin-card">
      <div className={`sin-icon ${tone}`}>
        <Icon size={24} aria-hidden="true" />
      </div>
      <h2>{title}</h2>
      <p>{description}</p>
      <div className="sin-example">
        <strong>Example:</strong>
        <span>{example}</span>
      </div>
    </article>
  );
}
