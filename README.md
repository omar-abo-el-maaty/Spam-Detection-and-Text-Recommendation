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
* 🤗 Live demo deployed on Hugging Face

---

## 🤗 Live Demo (Hugging Face)

Try the model directly from your browser:

🔗 https://huggingface.co/spaces/omaraboelmaaty/spam-detection-and-word-recommendation

> This interactive demo allows users to classify messages and get next-word recommendations in real time.

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
* Select the best model

### 4. Recommendation System

* Build unigram & bigram frequency models
* Predict next word using probability

---

## 📂 Project Structure

```id="9j4xra"
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

```bash id="d2p6od"
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the model

```bash id="6d49kk"
python spam_classifier.py
```

### Example Prediction

```python id="y5h2ct"
text = ["Free entry! Claim your prize now"]
prediction = model.predict(text)
print(prediction)
```

### Next Word Recommendation

```python id="7m4v2y"
complete_sentence("please confirm your")
```

---

## 📊 Model Performance

| Model  | Accuracy |
| ------ | -------- |
| Binary | High     |
| Count  | High     |
| TF-IDF | Best     |

> The best model is automatically selected and saved.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib

---

## 💡 Future Improvements

* Deploy as a web app (Flask / FastAPI)
* Use Deep Learning models (LSTM / Transformers)
* Improve recommendation with advanced language models
* Add real-time API

---

## 👨‍💻 Author

Omar Aboelmaaty

