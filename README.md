# logisitc-regression-model

This program is training a logistic regression model to predict the sentiment (positive or negative) of customer reviews. It first imports several libraries such as pandas, sklearn, and nltk, which are used for various tasks including data manipulation, feature extraction, and natural language processing. The program then defines several functions:

"clean(document)" preprocesses a given text by lowercasing it, eliminating contractions, spell-checking, tokenizing, and removing stop words.

"train_model()" reads in three data files containing customer reviews, preprocesses them using the "clean()" function, vectorizes them using TfidfVectorizer, splits the data into train and test sets, and trains a logistic regression model using the train set. The accuracy of the model is printed at the end.

"predict()" takes in a new review, preprocesses it using the "clean()" function, vectorizes it, and makes a sentiment prediction using the trained model.

"testing()" calls "train_model()" and "predict()" in sequence.