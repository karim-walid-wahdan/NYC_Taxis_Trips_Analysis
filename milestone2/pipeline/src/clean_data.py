import pandas as pd
from pandas.core.frame import DataFrame
from pandas import Series
import numpy as np
import datetime as dt 
import math 
from scipy import stats
from sklearn import preprocessing as pp
import requests
import csv
import os
import warnings

# Ignore all warnings (not recommended unless necessary)
warnings.filterwarnings("ignore")

# preprocess_dataset
def extract_data(file_name):
    return pd.read_csv(file_name)

def clean_column_names(df:DataFrame):
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

def convert_date_time(df : DataFrame):
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    return df 

def split_datetime_series(datetime_series):
    date_series = datetime_series.dt.date
    time_series = datetime_series.dt.time
    hour_series = datetime_series.dt.hour
    day_series = datetime_series.dt.day_name()
    return date_series, time_series, hour_series, day_series

def process_all_datetime_cols(df: DataFrame):
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        def assign_datetime_components(datetime_series):
            date_col, time_col, hour_col, day_col = split_datetime_series(datetime_series)
            df[col + "_date"] = date_col
            df[col + "_time"] = time_col
            df[col + "_hour"] = hour_col
            df[col + "_day"] = day_col
        assign_datetime_components(df[col])
    return df 

def split_location_series(location_series):
        return location_series.str.split(',').str[0]

def split_location_df(df:DataFrame):    
    df["pu_area"] = split_location_series(df.pu_location)
    df["do_area"] = split_location_series(df.do_location)
    return df


# handle_incorrect_data
def handle_duplicates(df: DataFrame, subset=None):
    df.drop_duplicates(subset=subset,inplace=True)
    return df

def replace_negative_with_absolute(df):
    numeric_columns = df.select_dtypes(include='number')
    numeric_columns[:] = np.abs(numeric_columns.values)
    df[numeric_columns.columns] = numeric_columns
    return df

def replace_non_zero_incorrect_values(feature:Series, value_to_replace):
    feature.update(feature.apply(lambda x: value_to_replace if (x != 0) and (not pd.isnull(x)) else x))
    return feature

def replace_values_in_dataframe(df:DataFrame, replacements= {'congestion_surcharge': 2.75,'mta_tax': 0.5}):
    for feature, value_to_replace in replacements.items():
        replace_non_zero_incorrect_values(df[feature], value_to_replace)
    return df

def ensure_proper_time_window(df:DataFrame,month=8,year=2019):
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_columns:
        df[col] = df[col].apply(lambda x: x.replace(month=month, year=year))
    return df

def get_trip_duration(df:DataFrame):
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds()
    return df 

def swap_times(row):
    if row['trip_duration'] < 0:
        row['lpep_pickup_datetime'], row['lpep_dropoff_datetime'] = row['lpep_dropoff_datetime'], row['lpep_pickup_datetime']
    return row

def switch_pickup_and_dropoff_if_needed(df:DataFrame):
    df = get_trip_duration(df)
    df = df.apply(swap_times, axis=1)
    df = get_trip_duration(df)  
    return df

def modify_passenger_count(df:DataFrame):
    conditions = [
        (df['passenger_count'] == 555),
        (df['passenger_count'] > 9),
        ((df['passenger_count'] >= 7) & (df['passenger_count'] <= 8))
    ]
    choices = [
        5,
        1,
        df['passenger_count'] - 3
    ]
    df['passenger_count'] = np.select(conditions, choices, default=df['passenger_count'])
    return df 

# Clean_data
def find_records_with_multiple_unknown_or_none(df, threshold=3):
    unknown_or_none_records = []

    for index, row in df.iterrows():
        count = 0
        for col in df.columns:
            if row[col] == "Unknown" or row[col] == "None" or row[col] == "Unknown,Unknown" or row[col] == "Unknown,NV" or row[col]==None or row[col] == np.NaN :
                count += 1
            if count >= threshold:
                unknown_or_none_records.append((index))
                break

    return unknown_or_none_records

def drop_useless_records(df:DataFrame):
    usless_records = find_records_with_multiple_unknown_or_none(df)
    df.drop(usless_records, inplace=True)
    return df 

def remove_unknowns(df:DataFrame,LOOKUP:DataFrame,lookup_table=False):
    df.payment_type.replace("Uknown", None, inplace=True)
    df.rate_type.replace("Unknown", None, inplace=True)
    df.trip_type.replace("Unknown", None, inplace=True)
    df.pu_location.replace("Unknown,Unknown", None, inplace=True)
    df.pu_location.replace("Unknown,NV", None, inplace=True)
    df.pu_area.replace("Unknown", None, inplace=True)
    df.do_location.replace("Unknown,Unknown", None, inplace=True)
    df.do_location.replace("Unknown,NV", None, inplace=True)
    df.do_area.replace("Unknown", None, inplace=True)

    if lookup_table:
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "payment_type", "original_value": "Uknown", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "rate_type", "original_value": "Unknown", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "pu_location", "original_value": "Unknown,Unknown", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "do_location", "original_value": "Unknown,Unknown", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "pu_location", "original_value": "Unknown,NV", "imputed_value": "Imputed like 'Unkown,Unkown'"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "do_location", "original_value": "Unknown,NV", "imputed_value": "Imputed like 'Unkown,Unkown'"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "trip_type", "original_value": "Unknown", "imputed_value": "NONE"}
    return df

def impute_with_value(feature:Series, value,LOOKUP:DataFrame,lookup_table=False):
    if lookup_table:
        entryIndex = LOOKUP[(LOOKUP.column_name == feature.name) & (LOOKUP.imputed_value == "NONE")].index[0]
        LOOKUP.at[entryIndex, "imputed_value"] = value
    return feature.fillna(value,inplace=True)

def impute_with_mode(feature:Series,LOOKUP:DataFrame,lookup_table=False):
    if lookup_table:
        entryIndex = LOOKUP[(LOOKUP.column_name == feature.name) & (LOOKUP.imputed_value == "NONE")].index[0]
        LOOKUP.at[entryIndex, "imputed_value"] = feature.mode()[0]
    return feature.fillna(feature.mode()[0],inplace=True)

def impute_by_frequency(feature:Series,LOOKUP:DataFrame,lookup_table=False):
    choices = feature.value_counts().index
    freqs = feature.value_counts(normalize=True)
    nullCount = feature.isna().sum()
    imputations = pd.Series(np.random.choice(choices, p=freqs, size=nullCount))
    imputations.index = feature.loc[feature.isna()].index
    feature.loc[feature.isna()] = imputations
    if lookup_table:
        entryIndex = LOOKUP[(LOOKUP.column_name == feature.name) & (LOOKUP.imputed_value == "NONE")].index[0]
        LOOKUP.at[entryIndex, "imputed_value"] = "Imputed By Frequency"
    return feature

def impute_by_group_mode(df:DataFrame, feature:Series, condition_column, condition_value,LOOKUP:DataFrame,lookup_table=False):
    if condition_value is None:
        # Handle the case where condition_value is None
        mode_series = df[df[condition_column].isna()][feature].mode()
    else:
        # Calculate the mode for the specified condition_value
        mode_series = df[df[condition_column] == condition_value][feature].mode()
    if not mode_series.empty:
        mode = mode_series[0]
        df[feature] = df[feature].fillna(mode)
    else:
        mode = feature.mode()[0]
        df[feature] = df[feature].fillna(0)
    if lookup_table:
        entryIndex = LOOKUP[(LOOKUP.column_name == feature) & (LOOKUP.imputed_value == "NONE")].index[0]
        LOOKUP.at[entryIndex, "imputed_value"] = f"imputed by the Mode {mode} of {feature} ={condition_value}"
    return df

def impute_missing(df:DataFrame,LOOKUP:DataFrame,lookup_table=False):
    if lookup_table:
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "payment_type", "original_value": "Unknown", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "passenger_count", "original_value": "NAN", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "store_and_fwd_flag", "original_value": "NAN", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "congestion_surcharge", "original_value": "NAN", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "extra", "original_value": "NAN", "imputed_value": "NONE"}
        LOOKUP.loc[len(LOOKUP)] = {"column_name": "ehail_fee", "original_value": "NAN", "imputed_value": "NONE"}
        
    df = impute_by_group_mode(df, 'passenger_count', 'trip_type', None,LOOKUP,True)
    
    df = impute_by_group_mode(df, 'payment_type', 'vendor', 'Creative Mobile Technologies, LLC',LOOKUP,True)
    df.payment_type.replace("Unknown", None, inplace=True)
    df = impute_by_group_mode(df, 'payment_type', 'vendor', 'VeriFone Inc.',LOOKUP,True)
    
    impute_with_mode(df.rate_type,LOOKUP,True)
    impute_with_mode(df.trip_type,LOOKUP,True)
    
    impute_by_frequency(df.store_and_fwd_flag,LOOKUP,True)
    impute_with_mode(df.congestion_surcharge,LOOKUP,True)
    impute_with_value(df.extra, 0,LOOKUP,True)
    impute_with_mode(df.ehail_fee,LOOKUP,True)
    
    impute_by_frequency(df.pu_location,LOOKUP,True)
    impute_by_frequency(df.do_location,LOOKUP,True)
    df["pu_area"] = split_location_series(df.pu_location)
    df["do_area"] = split_location_series(df.do_location)    
    
    return df

# Outliers
def count_feature_outliers(feature:Series,factor=2):
    Q1 = feature.quantile(0.25)
    Q3 = feature.quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * factor
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    return len(feature[feature<lower])+len(feature[feature>upper])

def count_data_frame_outliers(df:DataFrame, fator=2):
    numericalCols = df.select_dtypes(include=np.number).columns.tolist()
    result = []
    for col in numericalCols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - fator * iqr
        upper_bound = q3 + fator * iqr
        outliers = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers.sum()
        result.append({
            "column": col,
            "outlier_count": outlier_count,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lowerBound": lower_bound,
            "upperBound": upper_bound
        })
    return pd.DataFrame(result)

def impute_outliers_with_value(feature:Series, value,factor=2):

    q1 = feature.quantile(0.25)
    q3 = feature.quantile(0.75)
    iqr = q3 - q1

    lowerBound = q1 - factor*iqr
    upperBound = q3 + factor*iqr
    feature = feature.apply(lambda x: x if lowerBound <= x <= upperBound else value)
    return feature

def impute_outliers_with_freq(feature:Series,LOOKUP,factor=2):

    q1 = feature.quantile(0.25)
    q3 = feature.quantile(0.75)
    iqr = q3 - q1

    lowerBound = q1 - factor*iqr
    upperBound = q3 + factor*iqr
    
    feature = feature.apply(lambda x: x if lowerBound <= x <= upperBound else None)
    impute_by_frequency(feature,LOOKUP)
    return feature

def handle_outliers(df:DataFrame,LOOKUP):

    df.trip_distance = impute_outliers_with_freq(df.trip_distance,LOOKUP)
    df.fare_amount = impute_outliers_with_freq(df.fare_amount,LOOKUP)
    df.tip_amount = impute_outliers_with_freq(df.tip_amount,LOOKUP)
    df.tolls_amount = impute_outliers_with_value(df.tolls_amount, df[df.tolls_amount != 0].tolls_amount.mean())
    df.total_amount = df.tolls_amount + df.tip_amount + df.fare_amount + df.extra + df.improvement_surcharge + df.mta_tax + df.ehail_fee
    df.trip_duration = impute_outliers_with_freq(df.trip_duration,LOOKUP)
    df.lpep_dropoff_datetime = df.lpep_pickup_datetime +  pd.to_timedelta(df.trip_duration, unit='s')
    return df

def handle_zero_trips(df:DataFrame):
    zeroDurationTrips = df[(df.trip_duration == 0)].index
    df = df.drop(zeroDurationTrips)
    missingDistanceTrips = df[(df.trip_distance == 0) & (df.trip_duration > 300)].index
    df = df.drop(missingDistanceTrips)
    return df

# Encoding
def label_encode(feature :Series, lookup_table=None):
    feature_copy = feature.copy()
    label_encoder = pp.LabelEncoder().fit(feature_copy)
    feature_copy = label_encoder.fit_transform(feature_copy)
    if not (lookup_table is None):
        for label in label_encoder.classes_:
            lookup_entry = {}
            lookup_entry["column_name"] = feature.name
            lookup_entry["original_value"] = label
            lookup_entry["encoded_value"] = label_encoder.transform([label])[0]
            lookup_table.loc[len(lookup_table)] = lookup_entry
    return (feature_copy)

def get_freq_Values(feature:Series, count):
    return [x for x in feature.value_counts().sort_values(ascending=False).head(count).index]

def oneHotEncode(df:DataFrame, col, freqLabels):
    for label in freqLabels:
        df[col + '_' + label.lower()] = np.where(df[col] == label, True, False)
    return df

# Feature_engineering 
def add_is_weekend_column(df:DataFrame):
    # Map days of the week to a binary 'isWeekend' column
    df['is_weekend'] = df['lpep_pickup_datetime_day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    return df
def extract_first_two_elements(series,df):
    # Split each element by comma and extract the first two elements
    extracted_data = series.str.split(',', n=1).str[:2]
    
    # Combine the first two elements into a single string
    combined_data = extracted_data.str.join(', ')
    series_name = series.name +"_boroughs"
    df[series_name] = combined_data
    # Get unique values
    unique_combined_data = combined_data.unique()
    
    return unique_combined_data

def get_borough_coordinates(api_key, series,df ,csv_filename='coordinates.csv'):
    # Check if the CSV file already exists
    boroughs = extract_first_two_elements(series,df)
    if os.path.exists(csv_filename):
        print(f"{csv_filename} already exists. Returning existing data.")
        return pd.read_csv(csv_filename)
    base_url = "http://api.positionstack.com/v1/forward"
    coordinates = []
    for borough in boroughs:
        params = {
            "access_key": api_key,
            "query": borough
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if response.status_code == 200 and data.get('data'):
            location = data['data'][0]
            lat = location['latitude']
            lng = location['longitude']
            coordinates.append((borough, lat, lng))
    if coordinates:
        # Save the data to a CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Borough", "Latitude", "Longitude"])  # Write header
            for borough, lat, lng in coordinates:
                writer.writerow([borough, lat, lng])
        
        print(f"Data saved to {csv_filename}")
        return pd.read_csv(csv_filename)
    else:
        print("No data retrieved. CSV file not created.")
    return 
def add_coordinates(df, borough_column, coordinates_df,col_name):
    # Create dictionaries mapping boroughs to their respective coordinates
    borough_to_lat = dict(zip(coordinates_df['Borough'], coordinates_df['Latitude']))
    borough_to_long = dict(zip(coordinates_df['Borough'], coordinates_df['Longitude']))
    
    # Map the values using the dictionaries
    df[col_name+'_latitude'] = df[borough_column].map(borough_to_lat)
    df[col_name+'_longitude'] = df[borough_column].map(borough_to_long)
    
    return df

def add_features(df:DataFrame,api_key = '7c119672ad14840a3e94e0eb514c7305'):
    df = add_is_weekend_column(df)
    coor_pickup = get_borough_coordinates(api_key,df.pu_location,df,csv_filename='/app/data/coordinates_pickup.csv')
    coor_dropoff = get_borough_coordinates(api_key,df.do_location,df,csv_filename='/app/data/coordinates_dropoff.csv')
    df = add_coordinates(df, 'pu_location_boroughs', coor_pickup,"pu")
    df = add_coordinates(df, 'do_location_boroughs', coor_dropoff,"do")
    return df 
# Main_Functions
def preprocess_dataset(filename):
    df = extract_data(filename)
    df = clean_column_names(df)
    df = convert_date_time(df)
    df = process_all_datetime_cols(df)
    df = split_location_df(df)
    return df 

def handle_incorrect_data(df:DataFrame):
    df = handle_duplicates(df)
    df = replace_negative_with_absolute(df)
    df = replace_values_in_dataframe(df)
    df = ensure_proper_time_window(df)
    df = switch_pickup_and_dropoff_if_needed(df)
    df = modify_passenger_count(df)
    return df

def create_lookup ():
    return pd.DataFrame(columns=["column_name", "original_value", "encoded_value", "imputed_value"])

def clean_data(df:DataFrame,LOOKUP:DataFrame):
    df = drop_useless_records(df)
    df = remove_unknowns(df,LOOKUP,True)
    df = impute_missing(df,LOOKUP,True)
    return df

def manage_outliers(df:DataFrame,LOOKUP):
    df = handle_outliers(df,LOOKUP)
    df = handle_zero_trips(df)
    return df

def encode_dataFrame(df:DataFrame,LOOKUP):
    df = pd.get_dummies(df, columns=["mta_tax","ehail_fee","improvement_surcharge"], drop_first=True)
    df = oneHotEncode(df, "rate_type", get_freq_Values(df.rate_type, 3))
    df.store_and_fwd_flag = label_encode(df.store_and_fwd_flag, LOOKUP)
    df.vendor = label_encode(df.vendor, LOOKUP)
    df.trip_type = label_encode(df.trip_type, LOOKUP)
    df.payment_type = label_encode(df.payment_type, LOOKUP)
    return df 

def discretise_dates(df:DataFrame):
    min_date = df.lpep_pickup_datetime.min().date() - dt.timedelta(days=1)
    max_date = df.lpep_dropoff_datetime.max().date() + dt.timedelta(days=1)
    numberOfWeeks = math.ceil((max_date - min_date).days / 7)
    dateBins = pd.DatetimeIndex([min_date + dt.timedelta(weeks = i) for i in range(numberOfWeeks+1)])
    binLabels = [i+1 for i in range(len(dateBins)-1)]
    df["week_number"] =  pd.cut(x = df.lpep_pickup_datetime, bins=dateBins, labels=binLabels, include_lowest=False)
    df["date_range"] = pd.cut(x = df.lpep_pickup_datetime,bins=dateBins, include_lowest=False)
    return df

def normalise_trip_duration(df:DataFrame):
    df_copy = df.copy()
    df_copy.trip_duration = stats.boxcox(df.trip_duration)[0]
    return df_copy 