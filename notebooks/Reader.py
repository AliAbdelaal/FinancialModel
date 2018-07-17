import os
import pandas as pd

RES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'res', 'EURUSD_15m_BID_01.01.2010-31.12.2016.csv')

def read_data():
    df = pd.read_csv(RES_DIR)
    df['Time'] = pd.to_datetime(df.Time)
    df.set_index('Time', inplace=True)
    return df
