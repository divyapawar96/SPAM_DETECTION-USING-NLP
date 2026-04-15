<div align="center">

# 📧 Spam Email Detection using NLP 🛡️

**A robust machine learning application to automatically classify emails and messages as Spam or Not Spam (Ham) using Natural Language Processing.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-green?style=for-the-badge)]()
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-%23FE4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/divyapawar96/SPAM_DETECTION-USING-NLP)

</div>

---

## 📖 Project Description

With the exponential increase in digital communication, spam messages pose a significant productivity drain and security risk. This project tackles the challenge head-on by leveraging **Natural Language Processing (NLP)** and Machine Learning to automatically filter out malicious or unwanted messages. 

By applying **TF-IDF vectorization** to extract meaningful numerical features from raw text, the project trains a highly efficient **Multinomial Naive Bayes** classifier that achieves outstanding precision. The solution includes a complete, end-to-end pipeline from text preprocessing to model evaluation, topped off with a sleek interactive **Streamlit web interface** for real-time predictions.

---

## ✨ Features

- **🧹 Advanced Text Preprocessing:** Custom pipeline handling lowercasing, punctuation removal, and stopword filtering.
- **🔢 TF-IDF Feature Extraction:** Converts unstructured text into a robust mathematical representation, prioritizing significant words.
- **⚡ High-Performance ML Classifier:** Powered by a Multinomial Naive Bayes model designed specifically for high-speed text classification.
- **📊 Comparative Analysis:** Includes Logistic Regression as a comparative baseline.
- **🖥️ Interactive UI:** A real-time web application to instantly test model predictions with custom text inputs.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 🐍
- **Data Manipulation:** Pandas, NumPy
- **NLP & Preprocessing:** NLTK
- **Machine Learning:** Scikit-Learn
- **Web Framework:** Streamlit
- **Serialization:** Joblib

---

## 📂 Project Structure

```text
SPAM_DETECTION-USING-NLP/
│
├── data/                      # Dataset directory (auto-downloaded)
├── models/                    # Saved ML models (.pkl)
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/                       # Source Code
│   ├── data_loader.py         # Script to download and test the dataset
│   └── train.py               # ML Pipeline: Preprocess, train, evaluate, save
│
├── app.py                     # Streamlit User Interface
├── requirements.txt           # Required Python packages 
└── README.md                  # Project Documentation
```

---

## 🚀 Installation & Setup

Follow these steps to run the complete project on your local machine.

**1. Clone the repository**
```bash
git clone https://github.com/divyapawar96/SPAM_DETECTION-USING-NLP.git
cd SPAM_DETECTION-USING-NLP
```

**2. Create a virtual environment**
```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install required dependencies**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1️⃣ Train the Model via CLI
If you want to train the model from scratch and evaluate its performance, run the core training script. It will automatically download the required dataset, start preprocessing, and save the finalized `.pkl` models.
```bash
python src/train.py
```

### 2️⃣ Run the Interactive Web UI
To test the spam detector interactively, launch the Streamlit frontend:
```bash
streamlit run app.py
```
*(The web application will automatically open in your default browser at `http://localhost:8501`)*

---

## 🧪 Sample Input & Output

| Input Message | Model Prediction | Confidence Level |
| :--- | :---: | :---: |
| *"Hey, are we still meeting for the project review at 4 PM?"* | ✅ **NOT SPAM (Ham)** | 98.4% |
| *"URGENT! You've been selected for a $1000 Walmart Gift Card. Click here to claim NOW!"* | 🚨 **SPAM** | 99.2% |

---

## 📊 Model Performance

Evaluated on the benchmark SMS Spam Collection dataset (Test split of 20%):

* **Multinomial Naive Bayes (Primary Model):**
  * **Accuracy:** 97.13%
  * **Precision:** 100.00% *(Zero legitimate emails falsely classified as spam)*
  * **Recall:** 78.52%

* **Logistic Regression (Baseline):**
  * **Accuracy:** 96.05%
  * **Precision:** 97.30%
  * **Recall:** 72.48%

---

## 📸 Screenshots

*(Add your screenshots here)*

* **Home Screen:**
  ![App Home Placeholder](https://via.placeholder.com/800x400.png?text=Streamlit+App+Interface)

* **Prediction Result:**
  ![Spam Result Placeholder](https://via.placeholder.com/800x400.png?text=Spam+Classification+Result)

---

## 🔮 Future Improvements

- [ ] Add advanced NLP techniques like Lemmatization or Stemming.
- [ ] Implement robust deep learning architectures (LSTMs or fine-tuned BERT models).
- [ ] Build a fully functional standalone REST API via FastAPI.
- [ ] Cloud deployment to Streamlit Cloud or Heroku.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Divya Pawar**
- GitHub: [@divyapawar96](https://github.com/divyapawar96)

---
<div align="center">
Made with ❤️ by Divya Pawar
</div>
