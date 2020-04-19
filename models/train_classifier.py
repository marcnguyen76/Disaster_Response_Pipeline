import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(database):
    """
    Loads dataframe from database
    :param database: Name of database file
    :return:
        X: messages
        y: categories
        column_names: column names of key words
    """
    # Loads dataframe from database
    path = 'sqlite:///' + database
    engine = create_engine(path)
    df = pd.read_sql_table('DisasterResponse', con=engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    column_names = y.columns

    return X, y, column_names


def tokenize(text):
    """
    Tokenizes the data
    :return: clean tokenized data
    """
    # Replaces url with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for urls in detected_urls:
        text = text.replace(urls, "urlplaceholder")

    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = []
    clean_tokens = []
    for tok in tokenizer:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        tokens.append(clean_tok)
        clean_tokens = tokens

    return clean_tokens


def build_model():
    """
    Build model
    :return: cross validation from GridSearch
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model
    :return: Performance dataframe
    """
    # predict on the X_test
    y_pred = model.predict(X_test)

    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)
    return performances


def save_model(model, model_filepath):
    """
    Saves model as pickle
    :param model: name of model
    :param model_filepath: model pathname
    :return: none
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
