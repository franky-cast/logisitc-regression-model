# logisitc-regression-model

Logistic regression model that predicts the sentiment (positive or negative) of customer reviews. First imports several libraries such as pandas, sklearn, and nltk, which are used for various tasks including data manipulation, feature extraction, and natural language processing. Afterwards, defines several functions:

"clean(document)" preprocesses a given text by lowercasing it, eliminating contractions, spell-checking, tokenizing, and removing stop words.

"train_model()" reads in three data files containing customer reviews, preprocesses them using the "clean()" function, vectorizes them using TfidfVectorizer, splits the data into train and test sets, and trains a logistic regression model using the train set. The accuracy of the model is printed at the end.

"predict()" takes in a new review, preprocesses it using the "clean()" function, vectorizes it, and makes a sentiment prediction using the trained model.

"testing()" calls "train_model()" and "predict()" in sequence.

<img width="947" alt="Screenshot 2023-09-18 at 11 56 40 AM" src="https://github.com/franky-cast/logisitc-regression-model/assets/113398924/bf96d836-06f6-4828-bc26-282be69376f7">
