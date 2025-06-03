# Read olist, utlc_apps and utlc_movies from train and test datasets and count the number of rows in each dataset.

import pandas as pd

def count_rows_in_datasets():
    # Load datasets
    olist = pd.read_csv('./data/train/olist.csv')
    utlc_apps = pd.read_csv('./data/train/utlc_apps.csv')
    utlc_movies = pd.read_csv('./data/train/utlc_movies.csv')

    # Count rows in each dataset
    olist_count = len(olist)
    utlc_apps_count = len(utlc_apps)
    utlc_movies_count = len(utlc_movies)

    return olist_count, utlc_apps_count, utlc_movies_count

def count_rows_in_test_datasets():
    # Load test datasets
    olist_test = pd.read_csv('./data/test/olist.csv')
    utlc_apps_test = pd.read_csv('./data/test/utlc_apps.csv')
    utlc_movies_test = pd.read_csv('./data/test/utlc_movies.csv')

    # Count rows in each test dataset
    olist_test_count = len(olist_test)
    utlc_apps_test_count = len(utlc_apps_test)
    utlc_movies_test_count = len(utlc_movies_test)

    return olist_test_count, utlc_apps_test_count, utlc_movies_test_count

def count_rows_in_raw_datasets():
    # Load raw datasets
    olist_raw = pd.read_csv('./data/raw/olist.csv')
    utlc_apps_raw = pd.read_csv('./data/raw/utlc_apps.csv')
    utlc_movies_raw = pd.read_csv('./data/raw/utlc_movies.csv')

    # Count rows in each raw dataset
    olist_raw_count = len(olist_raw)
    utlc_apps_raw_count = len(utlc_apps_raw)
    utlc_movies_raw_count = len(utlc_movies_raw)

    return olist_raw_count, utlc_apps_raw_count, utlc_movies_raw_count

if __name__ == "__main__":

    print(f"\n{'='*40} Raw Datasets {'='*40}\n")
    olist_raw_count, utlc_apps_raw_count, utlc_movies_raw_count = count_rows_in_raw_datasets()
    print(f"Number of rows in olist raw: {olist_raw_count}")
    print(f"Number of rows in utlc_apps raw: {utlc_apps_raw_count}")
    print(f"Number of rows in utlc_movies raw: {utlc_movies_raw_count}")


    print(f"\n{'='*40} Train Datasets {'='*40}\n")
    olist_count, utlc_apps_count, utlc_movies_count = count_rows_in_datasets()
    print(f"Number of rows in olist: {olist_count}")
    print(f"Number of rows in utlc_apps: {utlc_apps_count}")
    print(f"Number of rows in utlc_movies: {utlc_movies_count}")

    print(f"\n{'='*40} Test Datasets {'='*40}\n")
    olist_test_count, utlc_apps_test_count, utlc_movies_test_count = count_rows_in_test_datasets()
    print(f"Number of rows in olist test: {olist_test_count}")
    print(f"Number of rows in utlc_apps test: {utlc_apps_test_count}")
    print(f"Number of rows in utlc_movies test: {utlc_movies_test_count}")

