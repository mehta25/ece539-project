import numpy as np
import pandas as pd 
import re
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

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

    return model, cv, le

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
    return model, cv, le

def main():
    lang_model, lang_cv, lang_le = languageClassifier("language.csv")
    trained_lang_sentiment_dict = {}
    while True:
        text_in = input("Enter text to classify: ")
        exit_codes = ["exit", "quit", "q", "e"]
        if text_in in exit_codes:
            break
        if text_in == "":
            continue
        lang_pred = classifyLanguage(text_in, lang_model, lang_cv, lang_le)
        if lang_pred not in ["English", "Spanish", "Hindi", "French", "Arabic"]:
            print("Language not supported")
            continue
        if lang_pred[0] not in trained_lang_sentiment_dict:
            trained_lang_sentiment_dict[lang_pred[0]] = trainSentiment(lang_pred[0])
        model, cv, le = trained_lang_sentiment_dict[lang_pred[0]]
        sentiment_pred = classifyLanguage(text_in, model, cv, le)
        print("\n-----------------------------------------------------\n")
        print("Language: " + lang_pred[0])
        print("Sentiment: " + ("Positive" if str(sentiment_pred[0]) == "1" else "Negative"))
        print("\n-----------------------------------------------------\n")


if __name__ == "__main__":
    main()