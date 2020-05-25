import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import time


# Extract the data
# create path
file_dir = 'C:/Users/Danie/Desktop/UT Data/Movies-ETL/'


# read json file
with open (f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw =json.load(file)



# create dateframes from csv
kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# Tranform the data


try:
# Wikipedia data
# filter out tv shows column
    wiki_movies = [movie for movie in wiki_movies_raw 
                   if ('Director' in movie or 'Directed by' in movie) 
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles
        
        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
            
        # change the column names
        change_column_name('Directed by', 'Director')
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

    # rerun out list comprehension to clean wiki_movies and recreate df

    clean_movies = [clean_movie(movie) for movie in wiki_movies]

    wiki_movies_df = pd.DataFrame(clean_movies)


    # extract the IMDb ID from the IMDb link with "str.ectract()"
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    # droping duplicates 
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)


    # removing columns with null values
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
    print ("Wikipedia columns cleaned!")
except:
    print("Wikipedia columns failed to clean!")

# start cleaning box office columns
# view rows where box office is defined. drop the missing values
try:
    box_office = wiki_movies_df['Box office'].dropna()

    # lambda x: type(x) != str
    box_office[box_office.map(lambda x: type(x) != str)]

    # use join() function only when data points are lists.
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    # Tranforming box office data to specified format
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    # remove any values between a dollar sign and hyphen
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

         # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan
    # add clean box office column
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    # remove old box office column
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    print("Box office data cleaned!")
except:
    print("Box office failed to clean!")

# clean budget column
try:

    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget[~matches_form_one & ~matches_form_two]

    # Remove the citation references
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]

    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    wiki_movies_df.drop('Budget', axis=1, inplace=True)
    print("Budget data cleaned!")
except:
    print("Budget failed to clean!")

# parse release date
try:
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'


    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # parse running time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    # start to convert strings to numeric
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    wiki_movies_df.drop('Running time', axis=1, inplace=True)
    print("Parsed release date and running time successfully!")
except:
    print("Failed to parse release date and running time!")

# kaggle data cleaning
try:
    # data types fixing
    # The following code will keep rows where the adult column is False, and then drop the adult column.
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # creates the Boolean column we want and to assign it back to video:
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # We’ll make sure the errors= argument is set to 'raise', so we’ll know if there’s any data that can’t be converted to numbers
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    # assign timestamp column
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    print("Kaggle data cleaned!")
except:
    print("Failed to clean Kaggle data!")

# merge wikipedia and kaggle metadate
try:
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')


    movies_df['video'].value_counts(dropna=False)



    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


    movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)
    print("Wikipedia and kaggle data successfully!")
except:
    print("Failed to merge Wikipedia and kaggle data!")

# transform and merge rating data
try:
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1).pivot(index='movieId',columns='rating', values='count')

    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # fill na with a zero
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    print(" Merge rating data successful!")
except:
    print("Failed to merge rating data!")

# Loading Pandas DataFrame into SQL Database

try:
    # Create Database Engine and SQL movie data table
    connection_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    engine = create_engine(connection_string)
    movies_df.to_sql(name="movies", con=engine, if_exists="replace")
    
    print("Load to SQL Successful!")
except:
    print("Failed to load into SQL!")

# ETL for Ratings Data

try:
    # Delete data from ratings table
    delete_statement = "DELETE FROM ratings"
    engine.connect().execute(delete_statement)
    # create a variable for the number of rows imported
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
        
        # print out the range of rows that are being imported
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        # increment the number of rows imported by the size of 'data'
        rows_imported += len(data)
        # print that the rows have finished importing
        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')
except:
    print("Failed to load Ratings Data into SQL")

