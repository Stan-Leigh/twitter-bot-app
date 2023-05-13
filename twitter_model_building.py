import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Random Forest
import pickle

midterm = pd.read_csv('C:/Users/ugonn/Videos/HackCity/twitter_human_bots_dataset.csv/midterm.csv')
cresci = pd.read_csv('C:/Users/ugonn/Videos/HackCity/twitter_human_bots_dataset.csv/cresci.csv')
kaggle = pd.read_csv('C:/Users/ugonn/Videos/HackCity/twitter_human_bots_dataset.csv/twitter_human_bots_dataset.csv', index_col=0)

# convert time columns to datetime objects
midterm['probe_timestamp'] = pd.to_datetime(midterm['probe_timestamp'])
midterm['user_created_at'] = pd.to_datetime(midterm['user_created_at'])
kaggle['created_at'] = pd.to_datetime(kaggle['created_at'])
cresci['created_at'] = pd.to_datetime(cresci['created_at'])

# create calculated column: account age
midterm['account_age_days'] = (midterm['probe_timestamp'] - midterm['user_created_at']).dt.days

# drop unrelated columns
midterm.drop(columns=['probe_timestamp', 'name', 'url', 'protected', 'listed_count', 'tid', 'tweet_ids'], inplace=True)

# drop columns not used in building the model
midterm.drop(columns=['user_id', 'screen_name', 'lang'], inplace=True)
kaggle.drop(columns=['default_profile_image', 'id', 'lang', 'location', 'profile_image_url', 'screen_name'], inplace=True)

kaggle.drop('average_tweets_per_day', axis=1, inplace=True)

# rename columns for both datasets so they have matching columns
midterm.rename(columns={'user_created_at': 'created_at', 
                        'profile_use_background_image': 'profile_background_image_url'}, inplace=True)

# Refactor profile_background_image_url before .concat to prevent problems I haven't seen yet
kaggle["profile_background_image_url"] = kaggle["profile_background_image_url"].notnull().astype("int")
midterm['profile_background_image_url'].replace([True, False], [1, 0], inplace=True)

# approximate the probe_timestamp column in order to calculate account_age_days
cresci['probe_timestamp'] = '2018-09-04'

# convert column to datetime object
cresci['probe_timestamp'] = pd.to_datetime(cresci['probe_timestamp'])

cresci['created_at'] = cresci['created_at'].dt.date
cresci['created_at'] = pd.to_datetime(cresci['created_at'])
cresci['account_age_days'] = (cresci['probe_timestamp'] - cresci['created_at']).dt.days

# drop columns not used in building the model
cresci.drop(columns=['default_profile_image', 'id', 'lang', 'location', 'profile_image_url',
                        'screen_name', 'probe_timestamp', 'tweet_ids'], inplace=True)

# Refactor profile_background_image_url
cresci["profile_background_image_url"] = cresci["profile_background_image_url"].notnull().astype("int")

# Join tables together
df = pd.concat([kaggle, midterm, cresci])

df['created_at_day'] = df.created_at.dt.day
df['created_at_month'] = df.created_at.dt.month
df['created_at_year'] = df.created_at.dt.year

# drop created_at column
df.drop('created_at', axis=1, inplace=True)

# Make columns Machine Learning ready
df['default_profile'].replace([True, False], [1, 0], inplace=True)
df["description"] = df["description"].notnull().astype("int")
df['geo_enabled'].replace([True, False], [1, 0], inplace=True)
df['verified'].replace([True, False], [1, 0], inplace=True)

df['account_type'].replace(['human', 'bot'], [1, 0], inplace=True)

df = df.drop_duplicates()

# Machine Learning
X = df.drop('account_type', axis=1)
y = df['account_type']

# create the model
rf = RandomForestClassifier(max_depth=110, max_features=2, n_estimators=200)

# fit the model
rf.fit(X, y)

# Saving the model
pickle.dump(rf, open('twitter_clf.pkl', 'wb'))
