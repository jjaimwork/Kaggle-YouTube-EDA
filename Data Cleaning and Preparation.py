# import libraries I need for everything
import os
import gc
import time
import re
from tqdm.notebook import tqdm as tqdm

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from wordcloud import WordCloud, ImageColorGenerator

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

# read datasets
us_df = pd.read_csv('USvideos.csv')
category_json = pd.read_json('US_category_id.json')

# drop unneeded  columns
us_df.drop(['video_id','thumbnail_link',
            'comments_disabled','ratings_disabled',
            'video_error_or_removed','description'],axis=1,inplace=True)

# get categorical id values
column_names = ['category_id','category']
catid_df = pd.DataFrame(columns=column_names)

for data in category_json['items']:
    category_id = data['id']
    category = data['snippet']['title']
    catid_df = catid_df.append({'category_id': category_id,
                                'category': category}, ignore_index=True)

catid_df['category_id'] = catid_df['category_id'].astype(int)
catid_df.set_index('category_id',inplace=True)
catid_df=catid_df.to_dict(orient='dict')

# replacing category id into it's desired value
us_df.replace({"category_id": catid_df['category']},inplace=True)
us_df.rename(columns={'category_id':'category'},inplace=True)

# changing format of Date Time
def conv_dates_series(df, col, old_date_format, new_date_format):

    df[col] = pd.to_datetime(df[col], format=old_date_format).dt.strftime(new_date_format)

    return(df)

old_date_format='%y.%d.%m'
new_date_format='%Y-%m-%d'

conv_dates_series(us_df, 'trending_date', old_date_format, new_date_format)

# creating different columns for different Months and Years
us_df['YYYY'] = us_df['trending_date'].apply(lambda x: x.split('-')[0])
us_df['MM'] = us_df['trending_date'].apply(lambda x: x.split('-')[1])

us_df.to_csv('us_df.csv', index=False)