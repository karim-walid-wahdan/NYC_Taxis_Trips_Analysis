
import clean_data as cd
import pandas as pd 
import os
from sqlalchemy import create_engine

data_dir = "/app/data/"
dataset_location = data_dir+"green_trip_data_2019-8_clean.csv"
lookup_location = data_dir+"lookup_table_green_taxis.csv"
if os.path.exists(dataset_location) and os.path.exists(lookup_location):
        print("the cleaned dataset and lookup already exists. Returning existing cleaned data and lookup table.", flush=True)
else :
    LOOKUP = cd.create_lookup()
    df = cd.preprocess_dataset(data_dir + "green_tripdata_2019-08.csv")
    df = cd.handle_incorrect_data(df)
    df = cd.clean_data(df,LOOKUP)
    df = cd.manage_outliers(df,LOOKUP)
    df = cd.encode_dataFrame(df,LOOKUP)
    df = cd.discretise_dates(df)
    df = cd.normalise_trip_duration(df)
    df = cd.add_features(df)
    LOOKUP.to_csv(lookup_location)
    df.to_csv(dataset_location)

dataset = pd.read_csv(dataset_location)
lookup = pd.read_csv(lookup_location)

engine = create_engine('postgresql://root:root@database:5432/green_taxi_8_2019_postgres')

try:
    connection = engine.connect()
    print('Connected successfully', flush=True)
    dataset.to_sql(name='green_taxi_8_2019', con=engine, if_exists='fail')
    lookup.to_sql(name='lookup_green_taxi_8_2019', con=engine, if_exists='fail')
    connection.close()  # Close the connection after use
except Exception as e:
    print('Failed to connect:',e, flush=True)