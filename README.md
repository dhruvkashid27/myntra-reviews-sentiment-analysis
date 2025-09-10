Myntra Reviews — Sentiment Analysis (NLP + ML)

Purpose:
Demonstrate how NLP + ML can classify e-commerce product reviews:
- Convert text to vectors (**TF-IDF**).
- Train & compare classifiers (**Naive Bayes, Logistic Regression, SVM, Random Forest**).
- Serve real-time predictions via **Streamlit**.

Outcomes:
- Modeling: Multi-model benchmark with accuracy & reports.
- App: Single review + CSV bulk prediction, downloadable results.
- Reproducibility: `train.csv` / `test.csv`, saved artifacts (`best_model.pkl`, `tfidf.pkl`).
  
Quickstart (Colab / Local):
```bash
pip install -r requirements.txt
streamlit run app.py
myntra-sentiment-analysis/
├─ app.py                  # Streamlit app (single + bulk prediction)
├─ model_training.ipynb    # Notebook: prep, TF-IDF, models, save artifacts
├─ train.csv               # Train split (Cleaned_Review, Sentiment)
├─ test.csv                # Test split  (Cleaned_Review, Sentiment)
├─ best_model.pkl          # Trained classifier (e.g., SVM)
├─ tfidf.pkl               # TF-IDF vectorizer
├─ requirements.txt
└─ README.md

Data:
Labels: Positive, Neutral, Negative
Columns: review (raw) → Cleaned_Review (processed), Sentiment
Run – Streamlit App
Single review
Open app → paste review → Predict Sentiment
Bulk CSV
Upload CSV with a review column
Get table with Predicted_Sentiment
Download annotated CSV
Train / Re-Train (optional)
Open model_training.ipynb and run all cells. Save artifacts:

import joblib
joblib.dump(best_model, "best_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

Example (Model Comparison)
Model               Accuracy
----------------------------
Naive Bayes         86.7%
Logistic Regression 90.0%
SVM                 92.5%
Random Forest       88.3%

Requirements:
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
joblib

Deploy (Streamlit Cloud):
Push repo to GitHub (include app.py, requirements.txt, best_model.pkl, tfidf.pkl)
Streamlit Cloud → New app → select repo → main file: app.py → Deploy
Get your public URL.

