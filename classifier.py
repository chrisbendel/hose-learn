# Classifier.py
# By Jonah Allibone
# Classifies Songs based on a user profile user a linear regression
from pymongo import MongoClient
import pprint
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow import logging
import tempfile
import sys
# Just disables the warning, doesn't enable AVX/FMA
import os
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

#sample size
SAMPLE_SIZE = 200
exported_path = './models/build/1521744594'
predictor = tf.contrib.predictor.from_saved_model(exported_path)

# PREDICT WITH MODEL
test_query = songs.aggregate(
   [ { '$sample': { "size": SAMPLE_SIZE } } ]
)

new_data = []
user_songs = historydb.find({"user_id": "5aae90ec037335016f8f6796"})
song_data = []
clustered_songs =[]

for song in historydb.find({"user_id": "5aae90ec037335016f8f6796"}):
  song_info = songs.find_one({"id": int(song["song_id"])})
  song_data.append(song_info)

for song_d in song_data:
  if(song_d):
    clustered_song = songs.find_one({"Cluster": song_d["Cluster"]})
    clustered_songs.append(clustered_song)

for c_song in clustered_songs:
  song_f = song_features.find_one({"Field_0": c_song["id"]})
  if song_f not in new_data:
    new_data.append(song_f)

# print(new_data)

test_data = []
iterator = 0

for data in test_query:
  merge = dict()
  features = song_features.find_one({'Field_0': int(data['id'])})
  merge.update(features)

  new_data.append(merge)
  iterator = iterator + 1

  sys.stdout.write('\r')
  sys.stdout.write('%.2f%% complete' % (iterator / SAMPLE_SIZE * 100,))
  sys.stdout.flush()


songsToExport = {}
final = []

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def predict_loop(song):  
  while not coord.should_stop():
    feature_dict = {}
    for feature in song:
      if(feature == '_id' or feature == 'Field_0'):
        pass
      else: 
        feature_dict[feature] = _float_feature(value=float(song[feature]))
    
    # Prepare model input
    model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    model_input = model_input.SerializeToString()

    # print(feature_dict)
    output_dict = predictor({"inputs": [model_input]})

    # print(" prediction Label is ", output_dict['classes'])
    # print('Probability : ' + str(np.argmax(output_dict['scores'])))
    if(np.argmax(output_dict['scores']) > 0):
      songsToExport.update({song['Field_0']: output_dict['scores']})
      final.append(song['Field_0'])
      recommendations.append(song)
      #End Thread
    coord.request_stop()
  print("done.")
 

with tf.Session() as sess:
  recommendations = []
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
  
  coord = tf.train.Coordinator()
  # get the predictor , refer tf.contrib.predictor
  threads = [Thread(target=predict_loop, args=(song,)) for song in new_data]
  
  for t in threads:
    t.start()
  coord.join(threads)

  print("recommendations count: " + str(len(recommendations)))
  # print(songsToExport)
  # print(final)
  # for recommendation in recommendations:
  #   print(recommendation['Field_0'])

for id in final:
  res = requests.get("http://phish.in/api/v1/tracks/" + str(id)).json()['data']
  print(res['title'] + " " + res['show_date'])