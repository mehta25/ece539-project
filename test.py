import numpy as np
import pandas as pd
import re
import sklearn.metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# visualize the results with a confusion matrix. use for both models
def plot_confusion_matrix(y_classified, y_true):
    # Compute confusion matrix
    c_mat = np.zeros((len(y_true),len(y_true)))
    for i in range(len(y_true)):
        c_mat[y_classified[i], y_true[i]] += 1
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=c_mat)
    disp.plot()


def languageClassifier(path):
    data = pd.read_csv(path)
    text = data['Text'].tolist()
    lang = data['Language'].tolist()
    for t in text:
        t = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', t)

    cv = CountVectorizer()
    print("CountVectorizer Initiated")
    X = cv.fit_transform(text).toarray()
    le = LabelEncoder()
    y = le.fit_transform(lang)
    print("LabelEncoder Initiated")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    print("Training and Testing Initiated")
    gnb = GaussianNB()
    print("GaussianNB Initiated")
    model = gnb.fit(X_train, y_train)
    print("Model Initiated")

    return model, cv, le, X_test, y_test

def classifyLanguage(text, model, cv, le):
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    X = cv.transform([text]).toarray()
    y_pred = model.predict(X)
    return le.inverse_transform(y_pred)

def trainSentiment(language):
    lang_codes = {"English": "en", "Spanish": "es", "Hindi": "hi", "French": "fr", "Arabic": "ar"}
    pos_text = pd.read_csv("positive_words_" + lang_codes[language] + ".txt", header=None).rename(columns={"index":"words"})
    neg_text = pd.read_csv("negative_words_" + lang_codes[language] + ".txt", header=None).rename(columns={"index":"words"})
    pos_text['Polarity'] = 1
    neg_text['Polarity'] = -1
    X = pd.concat((pos_text, neg_text), ignore_index=True)
    le = LabelEncoder()
    y = le.fit_transform(X['Polarity'])
    cv = CountVectorizer()
    X = cv.fit_transform(X[0])
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model = MultinomialNB().fit(X_train, y_train)
    return model, cv, le, X_test, y_test

def test_classify_lang():
    lang_model, lang_cv, lang_le, X_test, y_test = languageClassifier("language.csv")
    score = lang_model.score(X_test, y_test)
    print('     Language Classifier Accuracy Score: ' + str(score) + '\n')
    y_pred = lang_model.predict(X_test)
    #plot_confusion_matrix(y_pred, y_test)

# Method that takes language being tested, constructs model for the language, and prints scores
def test_sentiment(lang_to_test):
    sent_model, cv, le, X_test, y_test = trainSentiment(lang_to_test[0]) # sentiment model
    score = sent_model.score(X_test, y_test)
    print('     Sentiment Analysis Accuracy Score: ' + str(score) + '\n')
    y_pred = sent_model.predict(X_test)
    #plot_confusion_matrix(y_pred, y_test)

def main():
    lang_to_test = [["English"], ["Spanish"], ["Hindi"], ["French"], ["Arabic"]]
    print('Language Classification Testing Initiated...')
    test_classify_lang()
    print('Sentiment Analysis Testing Initiated...')
    print('Testing English:')
    test_sentiment(lang_to_test[0])
    print('Testing Spanish:')
    test_sentiment(lang_to_test[1])
    print('Testing Hindi:')
    test_sentiment(lang_to_test[2])
    print('Testing French:')
    test_sentiment(lang_to_test[3])
    print('Testing Arabic:')
    test_sentiment(lang_to_test[4])

if __name__ == "__main__":
    main()