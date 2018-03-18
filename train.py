# Train.py
# By Jonah Allibone
# Train model based on user listening data and song attributes

from pymongo import MongoClient
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

users = usersdb.find({}).limit(1)

song_features_dict = {"features": [], "labels": []}

base_nums = []

for key in song_features.find_one().keys():
  if key != '_id':
    base_nums.append(tf.feature_column.numeric_column(key))
    
song_features_dict["labels"] = np.array(base_nums)

def convert(arg):
  arg = tf.convert_to_tensor(arg)
  return arg

def convertToTensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg


def input_fn():
  for user in users:
    for song in user['likes']:
      training = song_features.find_one({"Field_0": song['song_id']})
      if training != None:
        del(training['_id'])
        song_features_dict["features"].append(training)

    
    song_features_dict["features"] = np.array(song_features_dict["features"])
    return convert(song_features_dict["labels"]), convert(song_features_dict["features"])
    # print(np.array(song_features_dict["labels"]))
    # print(np.array(song_features_dict["features"]))
    # print(convertToTensor(song_features_dict["labels"]))
    # print(convertToTensor(song_features_dict["features"]))
    # df = pd.DataFrame.from_dict(song_features_dict, orient='index').dropna()
    # matrix = df.as_matrix()
    # matrix = tf.convert_to_tensor(matrix)
    # print(matrix)
    # features = []
    # for value in matrix[0]:
    #   print(value)
    #   features.append(value)

    # labels = list(matrix)
    # features = convert(features)
    # labels = convert(labels)

    # print('features', features)
    # print('labels', labels)
    # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # return matrix
    # iterator = matrix.make_one_shot_iterator()
    # nums, labels = iterator.get_next()
    # return nums, labels

# print(base_nums)
model_dir = tempfile.mkdtemp()
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_nums)

model.train(input_fn=lambda: input_fn())
