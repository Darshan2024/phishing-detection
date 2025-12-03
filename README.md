# 🛡️ AI-Driven Multi-Layered Phishing Detection System  
**Machine Learning + Heuristics + Real-Time Streamlit Dashboard**

A complete phishing detection system that combines:
- **TF–IDF + Logistic Regression Machine Learning Model**
- **Heuristic rule-based detection engine**
- **Combined multi-layer risk scoring**
- **Interactive Streamlit dashboard**
- **Bulk analysis support**
- **Explainable output with heuristic breakdown**

This project demonstrates a practical proof-of-concept against modern phishing attacks.

---

## 📌 Features

### 🔍 1. Machine Learning Detection
- TF–IDF vectorization  
- Logistic Regression classifier  
- Predicts phishing probability from raw text  

### 🧠 2. Heuristic Analysis Engine
Detects:
- Suspicious keywords  
- Excessive capitalization / punctuation  
- Dangerous TLDs (`.tk`, `.ml`, `.ga`, `.ru`, `.cn`)  
- Multiple embedded links  
- Vague greetings (“Dear customer”)  

### 🔗 3. Combined Risk Scoring
Final Score = 0.6 × ML Probability + 0.4 × Heuristic Score

Outputs:
- 🔴 High Risk  
- 🟠 Medium Risk  
- 🟢 Low Risk  

### 🖥️ 4. Streamlit Dashboard
Includes:
- Single email analysis  
- Bulk email analysis (`---` separator)  
- Heuristic & ML breakdown  
- Probability meters  
- Clean UI  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/Darshan2024/phishing-detection.git
cd phishing-detection

2️⃣ Create Virtual Environment
python -m venv .venv

3️⃣ Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run Application
streamlit run app.py
