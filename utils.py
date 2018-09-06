import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


# useful constants
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'res', 'EURUSD_15m_BID_01.01.2010-31.12.2016.csv')

DAY_VALUES = 1
MONTH_VALUES = 30
WEEK_VALUES = DAY_VALUES*7
YEAR_VALUES = DAY_VALUES*365


def get_data(path=RES_DIR):
    df = pd.read_csv(path)
    return df


def generate_poly_feats(df, degree=3, y_col='next_rate'):
    df.dropna(inplace=True)
    for feature in df.columns:
        if feature == 'next_rate':
            continue
        poly_gen = PolynomialFeatures(degree=degree, include_bias=False)
        polys = poly_gen.fit_transform(df[[feature]])
        for column in range(degree):
            if column > 0:
                new_col_name = "{}^{}".format(feature, column+1)
                df[new_col_name] = polys[:, column]
    return df


def get_featured_data(path=RES_DIR):
    df = get_data(path)
    # set time as index
    df['Time'] = pd.to_datetime(df.Time)
    df = df.set_index('Time')

    # change the sampling rate to day for simplicity
    df = df.resample('D', convention='start').mean()

    # get time for further manipulation
    df['date'] = df.index.values

    # generate the average column to be used as indicator for the exchange rate
    df['Avg'] = (df['Low'] + df['High'])/2

    # time features (year, month, day)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)

    # Lagged Values (values over time)
    for unit, amount, shift_values in zip(['day', 'day', 'day', 'day', 'week', 'week', 'week', 'month', 'month', 'month', 'year'], [1, 2, 3, 4, 1, 2, 3, 1, 2, 1], [DAY_VALUES, DAY_VALUES, DAY_VALUES, DAY_VALUES, WEEK_VALUES, WEEK_VALUES, WEEK_VALUES, MONTH_VALUES, MONTH_VALUES, YEAR_VALUES]):
        for col in ['Open', 'Close', 'High', 'Low', 'Volume', 'Avg']:
            new_col = "{}_{}{}_before".format(col, amount, unit)
            df[new_col] = df[col].shift(amount*shift_values)

    # Summary of values (to get idea about how the value average over time)
    for unit, amount, win_size in zip(['day', 'day', 'week', 'week', 'month', 'month', 'month'], [1, 1, 1, 1, 1, 1, 1], [2, 5, 2, 3, 1, 2, 3]):
        for col in ['Open', 'Close', 'High', 'Low', 'Volume', 'Avg']:
            roll_col = "{}_av_{}{}_before_{}roll".format(
                col, amount, unit, win_size)
            shifted = "{}_{}{}_before".format(col, amount, unit)
            df[roll_col] = (df[shifted].rolling(window=win_size)).mean()

    # some stat of the values (max, min, avg)
    for col in ['Open', 'Close', 'High', 'Low', 'Avg']:
        window = df[col].expanding()
        df["{}_max".format(col)] = window.max()
        df["{}_min".format(col)] = window.min()
        df["{}_avg".format(col)] = window.mean()

    # drop the date column, we did already took all what we want from it
    df = df.drop("date", axis=1)

    # create the prediction column
    # where the exchange rate 1 if it goes up and -1 the other way around
    df['next_rate'] = np.where(df['Avg'].shift(-1) > df['Avg'], 1, -1)
    df = df.dropna()

    # scale the values for faster processing
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df
