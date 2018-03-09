# Train.py
# By Jonah Allibone
# Train model based on user listening data and song attributes

from pymongo import MongoClient
import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile
import sys


client = MongoClient('mongodb://hose:cawRXPhUuDT7uKPT@hose-shard-00-00-wndna.mongodb.net:27017,hose-shard-00-01-wndna.mongodb.net:27017,hose-shard-00-02-wndna.mongodb.net:27017/test?ssl=true&replicaSet=hose-shard-0&authSource=admin')
db = client['hose']

clusters = db.clusters
songs = db.song_collection
song_features = db.song_features
usersdb = db.user

# pprint.pprint(clusters.find({}))
users = usersdb.find({}).limit(1)

song_features_dict = {"features": [], "labels": []}

base_nums = []

for key in song_features.find_one().keys():
  if key != '_id':
    base_nums.append(tf.feature_column.numeric_column(key))


def convert(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

def input_fn():
  for user in users:
    progress = 0
    for song in user['likes']:
      training = song_features.find_one({"Field_0": song['song_id']})
      if training != None:
        del(training['_id'])
        # print(training)
        song_features_dict["features"].append(training)
      progress = progress + 1
      percent = progress/1000 * 100
      sys.stdout.write("progress: %d%%   \r" % (percent) )
      sys.stdout.flush()

    print("converting to dataframe")
    song_features_dict["labels"] = base_nums
    df = pd.DataFrame.from_dict(song_features_dict, orient='index')
    matrix = df.as_matrix()
    
    print("converted")
    # print(len(matrix))
    features = []
    print(matrix)
    for value in matrix[0]:
      print(value)

    labels = matrix[1]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    iterator = matrix.make_one_shot_iterator()
    nums, labels = iterator.get_next()
    return nums, labels

print(len(base_nums))
model_dir = tempfile.mkdtemp()
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_nums)

model.train(input_fn=lambda: input_fn())
