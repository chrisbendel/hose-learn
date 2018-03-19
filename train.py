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
song_features = db.songs
usersdb = db.user
historydb = db.play_history


def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg


training_data = []
training_labels = ["liked", "dislike", "skippped", "total_plays", "completed"]

for song in historydb.find():
  merge = dict()
  features = song_features.find_one({'Field_0': int(song['song_id'])})
  merge.update(features)
  merge.update(song)

  training_data.append(merge)

# print(training_data)

training_np = pd.DataFrame(training_data)

pprint.pprint(training_np)

num_columns = []
