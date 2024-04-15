from flask import Flask, render_template,request
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import _tree,DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 min_impurity=1e-7, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0,
                 tol=1e-4, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(seed=self.random_state)
        n_samples, n_features = X.shape
        self.estimators_ = []

        F_prev_train = np.zeros(n_samples)

        for i in range(self.n_estimators):
            residuals = y - self._sigmoid(F_prev_train)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_impurity_decrease=self.min_impurity,
                                          random_state=rng.integers(0, np.iinfo(np.int32).max))
            sample_weights = self.subsample * rng.choice(n_samples, n_samples, replace=True)
            sample_weights = sample_weights.astype(np.int64)  # Convert to int64 type
            tree.fit(X[sample_weights], residuals[sample_weights])
            self.estimators_.append(tree)

            F_train = F_prev_train + self.learning_rate * tree.predict(X)
            F_prev_train = F_train

            if i > 0 and np.abs(self._calculate_loss(y, F_train) - self._calculate_loss(y, F_prev_train)) < self.tol:
                break

    def predict_proba(self, X):
        F_ensemble = np.zeros(X.shape[0])

        for estimator in self.estimators_:
            F_ensemble += self.learning_rate * estimator.predict(X)

        proba = self._sigmoid(F_ensemble)
        return np.column_stack((1 - proba, proba))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calculate_loss(self, y_true, F_pred):
        proba = self._sigmoid(F_pred)
        return -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return accuracy, precision, recall, f1

# Define the MBTI definitions
mbti_definitions = {
    "ISTJ": "Reserved and practical, they tend to be loyal, orderly, and traditional.",
    "ISTP": "Highly independent, they enjoy new experiences that provide first-hand learning.",
    "ISFJ": "Warm-hearted and dedicated, they are always ready to protect the people they care about.",
    "ISFP": "Easy-going and flexible, they tend to be reserved and artistic.",
    "INFJ": "Creative and analytical, they are considered one of the rarest Myers-Briggs types.",
    "INFP": "Idealistic with high values, they strive to make the world a better place.",
    "INTJ": "Highly logical, they are both very creative and analytical.",
    "INTP": "Quiet and introverted, they are known for having a rich inner world.",
    "ESTP": "Out-going and dramatic, they enjoy spending time with others and focusing on the here-and-now.",
    "ESTJ": "Assertive and rule-oriented, they have high principles and a tendency to take charge.",
    "ESFP": "Outgoing and spontaneous, they enjoy taking center stage.",
    "ESFJ": "Soft-hearted and outgoing, they tend to believe the best about other people.",
    "ENFP": "Charismatic and energetic, they enjoy situations where they can put their creativity to work.",
    "ENFJ": "Loyal and sensitive, they are known for being understanding and generous.",
    "ENTP": "Highly inventive, they love being surrounded by ideas and tend to start many projects (but may struggle to finish them).",
    "ENTJ": "Outspoken and confident, they are great at making plans and organizing projects."
}

# Load the prefitted count vectorizer
with open("count_vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

# Load Random Forest models
rf_introv_model = pickle.load(open("rf_introv.pkl", "rb"))
rf_sens_model = pickle.load(open("rf_sens.pkl", "rb"))
rf_think_model = pickle.load(open("rf_think.pkl", "rb"))
rf_perc_model = pickle.load(open("rf_perc.pkl", "rb"))

# Load XGBoost models
xgb_introv_model = pickle.load(open("xgb_introv.pkl", "rb"))
xgb_sens_model = pickle.load(open("xgb_sens.pkl", "rb"))
xgb_think_model = pickle.load(open("xgb_think.pkl", "rb"))
xgb_perc_model = pickle.load(open("xgb_perc.pkl", "rb"))

# Load Logistic Regression models
lr_introv_model = pickle.load(open("lr_introv.pkl", "rb"))
lr_sens_model = pickle.load(open("lr_sens.pkl", "rb"))
lr_think_model = pickle.load(open("lr_think.pkl", "rb"))
lr_perc_model = pickle.load(open("lr_perc.pkl", "rb"))

# Load XGBoost scratch models
scratch_xgb_introv_model = pickle.load(open("scratch_xgb_introv.pkl", "rb"))
scratch_xgb_sens_model = pickle.load(open("scratch_xgb_sens.pkl", "rb"))
scratch_xgb_think_model = pickle.load(open("scratch_xgb_think.pkl", "rb"))
scratch_xgb_perc_model = pickle.load(open("scratch_xgb_perc.pkl", "rb"))

# Clean the input post
def clean_data(post):
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    ps = PorterStemmer()
    post = re.sub(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+)([\S])*", "", post)  # Remove links
    post = re.sub("\|\|\|", "", post)  # Remove |||
    post = re.sub("[0-9]", "", post)  # Remove numbers
    post = re.sub("[^a-z]", " ", post)  # Remove punctuation
    post = post.split()
    post = [word for word in post if word not in all_stopwords]  # Stopwords removal
    post = " ".join([ps.stem(word) for word in post])  # Stemming
    return post

# Predict MBTI type from the input post
def predict_mbti_type(post):
    post = post.lower()
    post = clean_data(post)
    post = cv.transform([post])
    logistic_prediction = [
        lr_introv_model.predict(post),
        lr_sens_model.predict(post),
        lr_think_model.predict(post),
        lr_perc_model.predict(post)
    ]
    rf_prediction = [
        rf_introv_model.predict(post),
        rf_sens_model.predict(post),
        rf_think_model.predict(post),
        rf_perc_model.predict(post)
    ]
    xgb_prediction = [
        xgb_introv_model.predict(post),
        xgb_sens_model.predict(post),
        xgb_think_model.predict(post),
        xgb_perc_model.predict(post)
    ]
    scratch_xgb_prediction = [
        scratch_xgb_introv_model.predict(post),
        scratch_xgb_sens_model.predict(post),
        scratch_xgb_think_model.predict(post),
        scratch_xgb_perc_model.predict(post)
    ]
    mbti_predictions = {
        "Logistic Regression": final_type(logistic_prediction),
        "Random Forest": final_type(rf_prediction),
        "XGBoost": final_type(xgb_prediction),
        "XGBoost Scratch": final_type(scratch_xgb_prediction)
    }
    return mbti_predictions

# Final MBTI type determination
def final_type(pred):
    mbti_types = [
        ["E", "I"],
        ["N", "S"],
        ["F", "T"],
        ["J", "P"]
    ]
    ans = []
    for i, p in enumerate(pred):
        ans.append(mbti_types[i][p[0]])
    return "".join(ans)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_ip = request.form['user_input']
    mbti_predictions = predict_mbti_type(user_ip)
    return render_template('result.html', mbti_predictions=mbti_predictions, mbti_definitions=mbti_definitions)

if __name__ == '__main__':
    app.run(debug=True)
