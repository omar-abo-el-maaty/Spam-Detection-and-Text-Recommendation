# 🚀 Smart Spam Filter & Text Recommendation System

An NLP-based machine learning project that classifies messages as **Spam or Ham** and provides **next-word recommendations** using a Bigram language model.

---

## 📌 Features

- 📩 Spam vs Ham classification using Machine Learning
- 🧹 Text preprocessing (cleaning, tokenization, stemming)
- 🔢 Multiple vectorization techniques:
  - Binary
  - Count Vectorizer
  - TF-IDF
- 🤖 Logistic Regression models comparison
- 📊 Model evaluation (Accuracy, Precision, Recall, F1-score)
- 🧠 Bigram-based next word recommendation system
- 💾 Model saving and loading using pickle
- ⚡ Served locally via a FastAPI app

---

## 📡 API

### 📍 Endpoint

```
POST /predict
```

### 📥 Example Request

```
{
  "text": "Free entry win prize now"
}
```

### 📤 Example Response

```
{
  "prediction": "Spam"
}
```

---

## 🧠 How It Works

### 1. Data Preprocessing

- Convert text to lowercase
- Remove URLs, punctuation, and numbers
- Remove stopwords
- Apply stemming

### 2. Feature Extraction

- Convert text into numerical format using:
  - CountVectorizer
  - TF-IDF

### 3. Model Training

- Train multiple Logistic Regression models
- Compare performance
- Select best model

### 4. Recommendation System

- Build unigram & bigram frequency model
- Predict next word using probability

---

## 📂 Project Structure

```
Spam-Detection-and-Text-Recommendation/
│
├── app.py
├── requirements.txt
├── README.md
├── NLP Project_v2.ipynb
│
├── models/
│   ├── best_spam_model.pkl
│   ├── best_vectorizer.pkl
│   └── combined_spam_recommendation_model.pkl
│
└── data/
    └── spam_ham_dataset.csv
```

---

## ⚙️ Installation

```
git clone https://github.com/omar-abo-el-maaty/Spam-Detection-and-Text-Recommendation.git
cd Spam-Detection-and-Text-Recommendation
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run locally

```
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### Example cURL Request

```
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Free entry win prize now"
  }'
```

---

## 📊 Model Performance

| Model  | Accuracy |
| ------ | -------- |
| Binary | High     |
| Count  | High     |
| TF-IDF | Best     |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- FastAPI

---

## 💡 Future Improvements

- Use Transformer models (BERT, GPT)
- Improve recommendation system
- Add real-time streaming API

---

## 👨‍💻 Author

**Omar Mohamed Ahmed Abo Elmaaty**

- GitHub: https://github.com/omaraboelmaaty
- Hugging Face: https://huggingface.co/omaraboelmaaty
