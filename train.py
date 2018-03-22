# Train.py
# By Jonah Allibone
# Train model based on user listening data and song attributes

from pymongo import MongoClient
import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import logging
import tempfile
import sys
import uuid
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.INFO)

# Mongo Client
client = MongoClient('mongodb://hose:cawRXPhUuDT7uKPT@hose-shard-00-00-wndna.mongodb.net:27017,hose-shard-00-01-wndna.mongodb.net:27017,hose-shard-00-02-wndna.mongodb.net:27017/test?ssl=true&replicaSet=hose-shard-0&authSource=admin')
# Mongo Database
db = client['hose']

# Clusters collection
clusters = db.clusters
# Song information collection
songs = db.song_collection
# Song features collection
song_features = db.songs
# Users collection
usersdb = db.user
# User play history collection
historydb = db.play_history

# Empty training data set
training_data = []

# Get songs for user
for song in historydb.find():
  merge = dict()
  features = song_features.find_one({'Field_0': int(song['song_id'])})
  if(features == None): 
    None
  else:
    merge.update(features)
    merge.update(song)
    training_data.append(merge)


# Make dataframe
training_np = pd.DataFrame(training_data)


# All columns including likes
# Type: TensorFlow column

base_columns = []
# All columns excluding likes
# Type: TensorFlow colum

feature_columns = []
# Names of all columns as strings
# Type: Strings
column_names = []

# Clean Columns
training_np = training_np.drop(['_id', 'song_id', 'user_id', 'Field_0', 'completed', 'dislike','skipped','total_plays'], axis=1)
# training_np[['like', 'completed', 'dislike']] = training_np[['like', 'completed', 'dislike']].astype(float)
training_np[['like']] = training_np[['like']].astype(float)

# Create base column names
for column in training_np:    
  base_columns.append(tf.feature_column.numeric_column(column))
  column_names.append(column)

# Set the feature columns
feature_columns = base_columns

# Remove Likes
feature_columns.pop(70)
for feature in feature_columns: 
  print(feature)
  print(len(feature_columns))
# Remove likes from column names
column_names.pop(70)

# Like LABEL
LABEL = 'like'


def my_input_fn(data_set):
  # Get feature values
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in column_names}

  # Get like values
  labels = tf.constant(data_set[LABEL].values)
  
  # print(feature_cols)
  # Return labels and feature cols
  return feature_cols, labels


model_dir = './models/tmp/' + str(uuid.uuid4().hex)
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=feature_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.5,
        l1_regularization_strength=5,
        l2_regularization_strength=0.1))


# Train Model 

model.train(input_fn=lambda: my_input_fn(training_np), steps=10000)

feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
servable_model_dir = "./models/build/"
servable_model_path = model.export_savedmodel(servable_model_dir, export_input_fn)
print("Done Exporting at Path - %s", servable_model_path )

