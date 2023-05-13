import pickle
import pandas as pd
import numpy as np
import tweepy
import json
import datetime

def prediction(username):
    # Get user data using Twitter API
    # keys and token to access the API
    consumer_key = 'consumer_key'
    consumer_secret = 'consumer_secret'
    access_token = 'access_token'
    access_secret = 'access_secret'

    # access the API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True)

    # function that gets data on user in json format
    def jsonify_tweepy(tweepy_object):
        # Write: Transform the tweepy's json object and transform into a dictionary
        json_str = json.dumps(tweepy_object._json, indent=2)
        # Read: Transform the json into a Python Dictionary
        return json.loads(json_str)

    user_data = api.get_user(screen_name=username)

    tweet_info = [{'default_profile': jsonify_tweepy(user_data)['default_profile'],
                'description': jsonify_tweepy(user_data)['description'],
                'favourites_count': jsonify_tweepy(user_data)['favourites_count'],
                'followers_count': jsonify_tweepy(user_data)['followers_count'],
                'friends_count': jsonify_tweepy(user_data)['friends_count'],
                'geo_enabled': jsonify_tweepy(user_data)['geo_enabled'],
                'profile_background_image_url': jsonify_tweepy(user_data)['profile_use_background_image'],
                'statuses_count': jsonify_tweepy(user_data)['statuses_count'],
                'verified': jsonify_tweepy(user_data)['verified'],
                'created_at': jsonify_tweepy(user_data)['created_at']
                }]

    df = pd.DataFrame(tweet_info)

    df['probe_timestamp'] = datetime.datetime.now()
    df['probe_timestamp'] = pd.to_datetime(df['probe_timestamp'])
    df['created_at'] = pd.to_datetime(df['created_at'])

    # create new columns
    df['account_age_days'] = (df['probe_timestamp'].dt.date - df['created_at'].dt.date).dt.days
    df['created_at_day'] = df.created_at.dt.day
    df['created_at_month'] = df.created_at.dt.month
    df['created_at_year'] = df.created_at.dt.year

    # drop columns not used in the model
    df.drop(columns=['probe_timestamp', 'created_at'], axis=1, inplace=True)

    # Make the required columns Machine Learning ready
    df['default_profile'].replace([True, False], [1, 0], inplace=True)
    df["description"] = df["description"].notnull().astype("int")
    df['geo_enabled'].replace([True, False], [1, 0], inplace=True)
    df['verified'].replace([True, False], [1, 0], inplace=True)
    df['profile_background_image_url'].replace([True, False], [1, 0], inplace=True)

    # Reads in saved classification model
    load_clf = pickle.load(open('twitter_clf.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)

    # Model precentage prediction
    prediction_proba = load_clf.predict_proba(df)

    account_type = np.array(['Bot','Human'])
    return account_type[prediction], prediction_proba[:, prediction]