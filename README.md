# 🎥 IMDb Movie Review Sentiment Analysis - Classification Problem

**IMDb-Movie-Review-Sentiment-Analysis** is a Python-based project designed to classify **movie reviews** as positive or negative using classification techniques. It leverages **data preprocessing, feature engineering, exploratory data analysis (EDA), and machine learning models** to perform sentiment analysis based on text reviews. The project addresses challenges like class imbalance and text vectorization to improve model accuracy.

---

## 📊 Dataset

The **ACL IMDb Dataset** is sourced from [Stanford AI](http://ai.stanford.edu/~amaas/data/sentiment/) and contains **50,000 movie reviews** split into training and test sets (25,000 each). Reviews are labeled as **positive** or **negative**:

- **Training Set**: 12,500 positive and 12,500 negative reviews.
- **Test Set**: 12,500 positive and 12,500 negative reviews.
- 
The dataset requires extraction from a tar.gz file and loading text files into a structured format for processing.

### 📚 Features Description

| Feature            | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| `Review Text`      | The raw text of the movie review (e.g., "This movie was great!").   |
| `Label`            | Binary label: 1 for positive, 0 for negative.                       |

The dataset involves text preprocessing to handle tasks like tokenization, stop-word removal, and vectorization (e.g., TF-IDF or Bag-of-Words).

---

## 💻 Code Structure

The project is implemented in a Jupyter Notebook (`reviews.ipynb`) with the following workflow:

1. **Data Loading**: Download and extract the ACL IMDb dataset using `wget` and `tar`.
2. **EDA**: Inspect dataset structure, check for duplicates, and analyze review lengths or word distributions (if implemented).
3. **Preprocessing**:
   - Load reviews from text files into DataFrames.
   - Handle any missing or corrupted data.
   - Vectorize text using techniques like TF-IDF (assumed, based on typical workflows).
4. **Feature Engineering**:
   - Convert text to numerical features (e.g., via `TfidfVectorizer`).
   - Balance classes if needed.
5. **Modeling**:
   - **Logistic Regression**: A linear model for binary classification, effective for high-dimensional text data.
   - **K-Nearest Neighbors (KNN)**: A distance-based classifier for identifying similar reviews.
   - **Random Forest**: An ensemble model using decision trees, robust to overfitting.
   - **XGBoost Classifier**: Tuned for high performance on imbalanced or noisy text data.
6. **Evaluation**: Use precision, recall, F1-score, and confusion matrix to assess model performance.

---

## 🚀 Key Features

- Classifies **movie reviews** as positive or negative using text-based models.
- Performs **data preprocessing** (text cleaning, vectorization).
- Conducts **EDA** with potential visualizations (e.g., word clouds, label distribution).
- Implements **feature engineering** (e.g., TF-IDF vectorization).
- Evaluates models with **precision, recall, F1-score**, and confusion matrix.
- Saves results for easy analysis and comparison.

---

## 📂 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/imdb-sentiment-analysis.git
   ```

2. Run the Jupyter Notebook:

   ```bash
   jupyter notebook reviews.ipynb
   ```

3. Check the output:
   - Model evaluations (e.g., classification reports, confusion matrices) are displayed in the notebook.
   - Processed dataset and predictions are available in the notebook environment.

---

## 📌 Why This Project?

Manually analyzing movie reviews for sentiment is **time-consuming and subjective**. This project **automates sentiment classification**, providing a streamlined workflow for data scientists, NLP enthusiasts, and movie analysts to **predict review polarity** and **understand key textual patterns** influencing positive/negative sentiments.

---

## 🙏 Acknowledgments

This project was developed as part of an AI or NLP learning exercise.  
Special thanks to the Stanford AI team for providing the ACL IMDb dataset.

---

## 🔗 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/ehsan-samy/)
- 📧 [Gmail](mailto:ehsansamy9@gmail.com)
- 🗃️ [Kaggle](https://www.kaggle.com/ehsansamy)
