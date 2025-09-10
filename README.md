Myntra Reviews — Sentiment Analysis (NLP + ML)

Overview:
This project analyzes Myntra product reviews and classifies them as Positive, Neutral, or Negative using TF-IDF vectorization and multiple ML models.
It also includes an interactive Streamlit app for real-time predictions and bulk CSV uploads.

Features:
✅ Text preprocessing & TF-IDF
✅ Multiple model training (Naive Bayes, Logistic Regression, SVM, Random Forest)
✅ Model comparison & evaluation metrics
✅ Streamlit App for real-time and bulk predictions
✅ Train/Test CSVs included for reproducibility

Project Structure:
myntra-sentiment-analysis/
│
├── app.py                  # Streamlit App
├── model_training.ipynb    # Colab Notebook
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── best_model.pkl          # Saved best model
├── tfidf.pkl               # Saved TF-IDF vectorizer
├── requirements.txt        # Dependencies
└── README.md               # Documentation


Tech Stack:
Python (Pandas, NumPy)
Scikit-learn (TF-IDF, ML classifiers, metrics)
Streamlit (UI)
Seaborn / Matplotlib / WordCloud (visualizations)
Joblib (model persistence)

Run Locally:
git clone https://github.com/<your-username>/myntra-sentiment-analysis.git
cd myntra-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py

Deploy on Streamlit Cloud:
Push repo to GitHub (include app.py, model files, and requirements.txt)
Go to Streamlit Cloud → New App
Select repo + app.py → Deploy 🚀
App will be live at:
https://<your-app-name>-<your-username>.streamlit.app

Example Workflow:
Single Review → Paste a review → Predict sentiment instantly
CSV Upload → Upload file with review column → Get predictions + downloadable results
Model Performance → Confusion matrix & accuracy metrics

Requirements:
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
joblib

Screenshots:
(Add screenshots of your Streamlit app UI, accuracy chart, and confusion matrix here)

Future Improvements:
Add BERT / Transformer models for higher accuracy
Add LIME/SHAP for explainability
Deploy on Heroku / Render for alternatives to Streamlit Cloud

