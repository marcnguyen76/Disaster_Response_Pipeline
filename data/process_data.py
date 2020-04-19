import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function reads data from specified filepath
    :param messages_filepath: Name of message csv file
    :param categories_filepath: Name of categories csv file
    :return: dataframe of merged files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the data by expanding the categories df, creating column names, and convert
    data text to int values
    :param df: Merged dataframe
    :return: Cleaned version of dataframe
    """
    # Split category data and creates column name and convert text to int
    categories_data = df['categories'].str.split(pat=';', expand=True)
    row = categories_data.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories_data.columns = category_colnames

    for column in categories_data:
        categories_data[column] = categories_data[column].str[-1:]
        categories_data[column] = categories_data[column].astype(int)

    # Filter data that are not one or zero
    for column in categories_data:
        categories_data = categories_data[categories_data[column] < 2]

    # Merge new data
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories_data], join='inner', axis=1, sort=False)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database):
    """
    Takes the dataframe and saves as SQL database
    :param df: Name of dataframe
    :param database: Name of SQL database
    :return:
    """
    database_name = 'sqlite:///' + database
    database_engine = create_engine(database_name)
    df.to_sql('Disasters', database_engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
