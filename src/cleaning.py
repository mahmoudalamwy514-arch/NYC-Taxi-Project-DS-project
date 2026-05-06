import pandas as pd

def clean_data(df):

    # remove duplicates
    df = df.drop_duplicates()

    # handle missing values
    df = df.dropna()

    # convert datetime
    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # remove invalid trip duration
    if "trip_duration" in df.columns:
        df = df[df["trip_duration"] > 0]

    # optional sanity checks (coordinates)
    df = df[
        (df["pickup_latitude"].between(-90, 90)) &
        (df["dropoff_latitude"].between(-90, 90)) &
        (df["pickup_longitude"].between(-180, 180)) &
        (df["dropoff_longitude"].between(-180, 180))
    ]

    print("After Cleaning:", df.shape)
    return df