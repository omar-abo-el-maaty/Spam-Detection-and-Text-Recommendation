# 🚀 Smart Spam Filter & Text Recommendation System

An NLP-based machine learning project that classifies messages as **Spam or Ham** and provides **next-word recommendations** using a Bigram language model.

---

## 📌 Features

* 📩 Spam vs Ham classification using Machine Learning
* 🧹 Text preprocessing (cleaning, tokenization, stemming)
* 🔢 Multiple vectorization techniques:

  * Binary
  * Count Vectorizer
  * TF-IDF
* 🤖 Logistic Regression models comparison
* 📊 Model evaluation (Accuracy, Precision, Recall, F1-score)
* 🧠 Bigram-based next word recommendation system
* 💾 Model saving and loading using pickle
* 🤗 Deployed on Hugging Face Spaces

---

## 🚀 Live Demo (Hugging Face)

### 🌐 Base URL

```
https://huggingface.co/spaces/omaraboelmaaty/spam-detection-and-word-recommendation
```

👉 You can test the model directly from the browser (no setup required).

---

## 📡 API (For Developers)

You can also use the model programmatically.

### 📍 Endpoint

```
POST /predict
```

### 🌐 Base URL

```
https://omaraboelmaaty-spam-detection-and-word-recommendation.hf.space
```

---

### 📥 Example Request

```json
{
  "text": "Free entry win prize now"
}
```

---

### 📤 Example Response

```json
{
  "prediction": "Spam"
}
```

---

## 🧠 How It Works

### 1. Data Preprocessing

* Convert text to lowercase
* Remove URLs, punctuation, and numbers
* Remove stopwords
* Apply stemming

### 2. Feature Extraction

* Convert text into numerical format using:

  * CountVectorizer
  * TF-IDF

### 3. Model Training

* Train multiple Logistic Regression models
* Compare performance
* Select best model

### 4. Recommendation System

* Build unigram & bigram frequency model
* Predict next word using probability

---

## 📂 Project Structure

```
project/
│
├── spam_classifier.py
├── requirements.txt
├── README.md
├── .gitignore
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

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run locally

```bash
python spam_classifier.py
```

---

### Example Prediction

```python
text = ["Free entry! Claim your prize now"]
prediction = model.predict(text)
print(prediction)
```

---

### Next Word Recommendation

```python
complete_sentence("please confirm your")
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

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Hugging Face Spaces

---

## 💡 Future Improvements

* Deploy as Flask/FastAPI backend
* Use Transformer models (BERT, GPT)
* Improve recommendation system
* Add real-time streaming API

---

## 👨‍💻 Author

Omar Aboelmaaty

