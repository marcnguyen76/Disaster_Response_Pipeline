# Disaster Response Pipeline Project

### Info:
In this project, I applied data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.  Included is a data set containing real messages that were sent during diaster events.  From the data, a machine learning pipeline is used to categorize these events so that you send the messages to an appropriate disaster relief agency.

A web app is included in the project, where an emergency worker can input a new message and get a classification results in several categories. The web app will also display visualizations of the data.

### Project Components:
1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
Web app that categories messages and provides visualization of the count of each category and the top five categories and its values. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
