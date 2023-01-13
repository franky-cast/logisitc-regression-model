import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from autocorrect import Speller
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

tfidf_vectorizer = None
model = None

# cleans up the text in the document that will be used for training the algorithm
def clean(document):
    # lowercasing
    lowercased_document = document.lower()

    # eliminating contraction words
    normalized_document_1 = lowercased_document.replace("hadn't", "had not")
    normalized_document_2 = normalized_document_1.replace("wasn't", "was not")
    normalized_document_3 = normalized_document_2.replace("didn't", "did not")

    # spell-checking
    spell = Speller()
    spell_checked_document = spell(normalized_document_3)

    # tokenizes document for stop word removal
    document_unigrams = word_tokenize(spell_checked_document)

    # removes stop words
    stop_words = stopwords.words('English')            
    document_no_stopwords = ' '.join([word for word in document_unigrams if word not in stop_words and word.isalnum() == True])

    # fully cleaned document
    cleanedDocument = document_no_stopwords

    return cleanedDocument


def train_model():
    global model
    global tfidf_vectorizer

    # reading the data files containing customer reviews
    amazon_data = pd.read_table('reviews/amazon_cells_labelled.txt', names=["review", "sentiment"])
    imdb_data = pd.read_table('reviews/imdb_labelled.txt', names=["review", "sentiment"])
    yelp_data = pd.read_table('reviews/yelp_labelled.txt', names=["review", "sentiment"])

    # combining thee three data files into 1
    data_original = pd.concat([amazon_data, imdb_data, yelp_data], ignore_index = True)

    # making copy of original, raw data
    data = data_original.copy()
    
    # preprocessing copy by calling clean()
    data.review = data.review.apply(clean)
    
    tfidf_vectorizer = TfidfVectorizer()

    # turning data into numeric form with tfidf_vectorizer's method
    data_numeric_form = tfidf_vectorizer.fit_transform(data["review"])

    # dividing data into train and test splits
    x_train, x_test, y_train, y_test = train_test_split(data_numeric_form.todense(), data["sentiment"])

    # instantiating an object (machine learning algorithm) from LogisticRegression()
    log_reg = LogisticRegression()
    model = log_reg

    # injecting the train split data into model (training)
    log_reg.fit(np.asarray(x_train), y_train)

    # determining the accuracy of  model (testing)
    accuracy = log_reg.score(np.asarray(x_test), y_test)
    print("Logistic Regression accuracy: ", accuracy)

def predict(review):
    global model
    global tfidf_vectorizer
    
    # cleaning  new review that we will predict
    new_review = clean(review)

    # turning it into numeric form
    # review_numeric_form = tfidf_vectorizer.transform([new_review])
    review_numeric_form = tfidf_vectorizer.transform([new_review])

    # making a prediction on this new review
    prediction = model.predict(review_numeric_form)

    print("The prediction for review: ", review, "is: ", prediction)
    
       
def testing():
    train_model()
    predict("This item is great!")
    
testing()