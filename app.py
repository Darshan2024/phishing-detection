# app.py
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

# Paths

MODEL_PATH = Path("email_model.pkl")  # your trained TF-IDF + LogisticRegression pipeline


# Load ML model

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None


model = load_model()


# Heuristic Engine

SUSPICIOUS_KEYWORDS = [
    "verify your account",
    "update your account",
    "suspend your account",
    "your account will be closed",
    "confirm your password",
    "click here",
    "login here",
    "log in here",
    "urgent action required",
    "you have won",
    "congratulations",
    "limited time",
    "act now",
    "paypal",
    "bank",
    "irs",
    "apple id",
    "amazon account",
    "security alert",
]

SUSPICIOUS_DOMAINS = [
    ".ru", ".cn", ".tk", ".ml", ".ga", ".gq"
]


def extract_links(text: str) -> List[str]:
    url_pattern = r"(https?://[^\s]+)"
    return re.findall(url_pattern, text, flags=re.IGNORECASE)


def heuristic_analysis(text: str) -> Dict:
    text_lower = text.lower()

    # 1. Suspicious keyword hits
    keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    keyword_score = min(len(keyword_hits) / 5.0, 1.0)

    # 2. Excessive punctuation / ALL CAPS
    exclamations = text.count("!")
    caps_words = [w for w in re.findall(r"\b[A-Z]{4,}\b", text) if w not in ("HTTP", "HTTPS")]
    style_score = min((exclamations / 5.0) + (len(caps_words) / 5.0), 1.0)

    # 3. Links / suspicious TLDs
    links = extract_links(text)
    suspicious_link_hits = 0
    for link in links:
        for dom in SUSPICIOUS_DOMAINS:
            if dom in link.lower():
                suspicious_link_hits += 1

    link_score = 0.0
    if links:
        link_score = 0.3 + min(suspicious_link_hits / 3.0, 0.7)
        link_score = min(link_score, 1.0)

    # 4. Vague greeting
    vague_greeting = any(
        pattern in text_lower
        for pattern in ["dear customer", "dear user", "dear valued", "dear sir", "dear madam"]
    )
    greeting_score = 0.4 if vague_greeting else 0.0

    # Overall heuristic score (0–1)
    weights = np.array([0.4, 0.2, 0.3, 0.1])
    components = np.array([keyword_score, style_score, link_score, greeting_score])
    overall_score = float(np.clip(np.dot(weights, components), 0.0, 1.0))

    return {
        "overall_score": overall_score,
        "keyword_score": float(keyword_score),
        "style_score": float(style_score),
        "link_score": float(link_score),
        "greeting_score": float(greeting_score),
        "keyword_hits": keyword_hits,
        "links": links,
        "vague_greeting": vague_greeting,
    }


# ML Prediction

def ml_predict_proba(text: str) -> Tuple[float, str]:
    """
    Returns (probability_phishing, label_str).

    Your saved model is a scikit-learn Pipeline (TF-IDF + LogisticRegression),
    so we can call model.predict_proba on raw text.

    NOTE: For consistency with your previous app, we assume:
        proba[0] -> Safe Email
        proba[1] -> Phishing Email
    """
    if model is None:
        return 0.0, "Model not loaded – using heuristics only"

    try:
        proba = model.predict_proba([text])[0]
        prob_safe = float(proba[0])
        prob_phish = float(proba[1])
    except Exception:
        # Fallback: if something weird happens, treat as very low phishing
        prob_safe = 0.9
        prob_phish = 0.1

    label = "Phishing Email" if prob_phish >= 0.5 else "Safe Email"
    return prob_phish, label


# Combined Analyzer

def combined_analysis(text: str) -> Dict:
    heur = heuristic_analysis(text)
    ml_prob, ml_label = ml_predict_proba(text)

    if model is None:
        combined_prob = heur["overall_score"]
    else:
        # Simple ensemble: ML (0.6) + Heuristics (0.4)
        combined_prob = float(0.6 * ml_prob + 0.4 * heur["overall_score"])

    if combined_prob >= 0.8:
        risk_label = "HIGH RISK (Treat as phishing)"
    elif combined_prob >= 0.5:
        risk_label = "MEDIUM RISK (Be very cautious)"
    else:
        risk_label = "LOW RISK (Likely legitimate, but still be careful)"

    return {
        "combined_prob": combined_prob,
        "risk_label": risk_label,
        "heuristics": heur,
        "ml_prob": ml_prob,
        "ml_label": ml_label,
    }


# UI Helpers

def risk_badge(score: float) -> str:
    if score >= 0.8:
        return "🔴 **HIGH RISK**"
    elif score >= 0.5:
        return "🟠 **MEDIUM RISK**"
    else:
        return "🟢 **LOW RISK**"


def show_score_meter(label: str, score: float):
    st.write(f"{label}: **{score:.2f}**")
    st.progress(min(max(score, 0.0), 1.0))


def analyze_single_email(text: str):
    if not text.strip():
        st.warning("Please paste an email first.")
        return

    result = combined_analysis(text)

    st.subheader("Overall Verdict")
    st.markdown(risk_badge(result["combined_prob"]))
    show_score_meter("Combined phishing probability", result["combined_prob"])
    st.caption(result["risk_label"])

    st.divider()

    col1, col2 = st.columns(2)

    # ML layer
    with col1:
        st.subheader("ML Model (TF-IDF + Logistic Regression)")
        if model is None:
            st.info("Model not loaded. Only heuristic engine is available.")
        else:
            show_score_meter("ML phishing probability", result["ml_prob"])
            st.write(f"Label: **{result['ml_label']}**")

    # Heuristic layer
    with col2:
        st.subheader("Heuristic Engine")
        heur = result["heuristics"]
        show_score_meter("Heuristic risk score", heur["overall_score"])

        with st.expander("Heuristic breakdown"):
            st.write(f"🔎 Keyword score: **{heur['keyword_score']:.2f}**")
            st.write(f"✨ Style score (caps/!): **{heur['style_score']:.2f}**")
            st.write(f"🔗 Link score: **{heur['link_score']:.2f}**")
            st.write(f"🙋 Greeting score: **{heur['greeting_score']:.2f}**")

            if heur["keyword_hits"]:
                st.write("Suspicious phrases found:")
                for kw in heur["keyword_hits"]:
                    st.markdown(f"- `{kw}`")
            if heur["links"]:
                st.write("Links found in email:")
                for link in heur["links"]:
                    st.markdown(f"- {link}")
            if heur["vague_greeting"]:
                st.write("⚠️ Vague greeting detected (e.g., *Dear customer*).")


# Bulk Analysis

def parse_bulk_input(raw: str) -> List[str]:
    """
    Separate multiple emails with a line containing exactly:
        ---
    """
    parts = [p.strip() for p in raw.split("\n---") if p.strip()]
    return parts


def bulk_analyze(raw: str):
    emails = parse_bulk_input(raw)
    if not emails:
        st.warning("Provide at least one email. Separate emails with a line containing `---`.")
        return

    rows = []
    for i, email in enumerate(emails, start=1):
        result = combined_analysis(email)
        rows.append(
            {
                "Email #": i,
                "Combined score": round(result["combined_prob"], 2),
                "Risk": result["risk_label"],
                "ML prob": round(result["ml_prob"], 2),
                "Heuristic score": round(result["heuristics"]["overall_score"], 2),
            }
        )

    st.subheader("Bulk Results")
    st.dataframe(rows, use_container_width=True)

    high_risk = [r for r in rows if r["Combined score"] >= 0.8]
    med_risk = [r for r in rows if 0.5 <= r["Combined score"] < 0.8]

    st.write(f"🔴 High risk emails: **{len(high_risk)}**")
    st.write(f"🟠 Medium risk emails: **{len(med_risk)}**")
    st.write(f"🟢 Low risk emails: **{len(rows) - len(high_risk) - len(med_risk)}**")


# Streamlit Layout

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Phishing Detection Dashboard")
st.caption("Multi-layer detection using ML model + heuristic rules + combined risk score.")

with st.sidebar:
    st.header("Tips")
    st.write("- Paste full email text (subject + body).")
    st.write("- For bulk mode, separate emails with a line containing `---`.")

tabs = st.tabs(
    [
        "🔍 Single Email Check",
        "📊 Bulk Check",
        "🧠 How It Works",
    ]
)

# ========== TAB 1: Single Email ==========
with tabs[0]:
    st.subheader("Analyze a Single Email")

    sample_toggle = st.checkbox("Use example suspicious email")

    default_text = ""
    if sample_toggle:
        default_text = (
            "Subject: Urgent – Verify your account now\n\n"
            "Dear Customer,\n\n"
            "We noticed suspicious activity in your bank account. "
            "To avoid suspension, please CLICK HERE and log in to verify your information.\n\n"
            "Failure to verify will result in immediate account closure.\n\n"
            "Best regards,\nSecurity Team\n"
            "https://secure-login-bank-verify.tk/login"
        )

    email_text = st.text_area(
        "Paste email content here (subject + body):",
        value=default_text,
        height=260,
    )

    if st.button("Run Analysis", type="primary"):
        analyze_single_email(email_text)

# ========== TAB 2: Bulk Check ==========
with tabs[1]:
    st.subheader("Bulk Email Analysis")

    st.markdown(
        "Paste multiple emails here. Separate each email with a line that contains **`---`** on its own."
    )

    bulk_example = (
        "Subject: Urgent – Verify your account now\n\n"
        "Dear Customer,\nPlease click the link below to verify your password.\n"
        "https://secure-verify-account.tk/\n\n"
        "---\n"
        "Subject: Team meeting tomorrow\n\n"
        "Hi all,\nJust a reminder that we have a team sync at 10am tomorrow.\n"
        "Best,\nManager"
    )

    use_example_bulk = st.checkbox("Use bulk example")
    bulk_text = st.text_area(
        "Bulk email input:",
        value=bulk_example if use_example_bulk else "",
        height=260,
    )

    if st.button("Analyze Bulk Emails"):
        bulk_analyze(bulk_text)

# ========== TAB 3: About / How It Works ==========
with tabs[2]:
    st.subheader("How This Detector Works")

    st.markdown(
        """
### 1. Machine Learning Layer
- Trained on **`Phishing_Email.csv`**.
- Uses a scikit-learn **Pipeline**: `TfidfVectorizer` + `LogisticRegression`.
- Given raw email text, the model outputs:
  - Probability that the email is **phishing**.
  - A label: **Phishing Email** or **Safe Email**.

### 2. Heuristic Layer
Rule-based checks that look for:
- Suspicious phrases (e.g., *verify your account*, *urgent action required*).
- Overuse of **ALL CAPS** and exclamation points.
- Links with unusual or high-risk domains (e.g. `.ru`, `.tk`).
- Vague greetings such as *Dear customer* instead of your real name.

These are combined into a heuristic risk score between 0 and 1.

### 3. Combined Probability
If the ML model is available:
- Final score = 0.6 × ML probability + 0.4 × heuristic score.
- Mapped to:
  - 🔴 **High risk**
  - 🟠 **Medium risk**
  - 🟢 **Low risk**
"""
    )
