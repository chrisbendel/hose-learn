# Classifier.py
# By Jonah Allibone
# Classifies Songs based on a user profile user a linear regression
from pymongo import MongoClient
import pprint
import pandas as pd
import numpy as np
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

exported_path = './models/build/1521663599'

#PREDICT WITH MODEL
test_query = songs.aggregate(
   [ { '$sample': { "size": 1000 } } ]
)

test_data = []
iterator = 0

for data in test_query:
  # print(data)
  
  merge = dict()
  features = song_features.find_one({'Field_0': int(data['id'])})
  merge.update(features)

  test_data.append(merge)
  iterator = iterator + 1

  sys.stdout.write('\r')
  sys.stdout.write('%.2f%% complete' % (iterator / 1000 * 100,))
  sys.stdout.flush()



def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def predict_loop(song):
    
  feature_dict = {}
  for feature in song:
    if(feature == '_id' or feature == 'Field_0'):
      print('')
    else: 
      feature_dict[feature] = _float_feature(value=float(song[feature]))

  # Prepare model input
  model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  model_input = model_input.SerializeToString()

  # get the predictor , refer tf.contrib.predictor
  predictor = tf.contrib.predictor.from_saved_model(exported_path)
  # print(feature_dict)
  output_dict = predictor({"inputs": [model_input]})

  # print(" prediction Label is ", output_dict['classes'])
  # print('Probability : ' + str(np.argmax(output_dict['scores'])))
  if(np.argmax(output_dict['scores']) > 0):
    recommendations.append(song)

  print("Thread finished")  

with tf.Session() as sess:
  recommendations = []
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
  
  coord = tf.train.Coordinator()

  threads = [Thread(target=predict_loop, args=(song,)) for song in test_data]
  
  for t in threads:
    t.start()
  coord.join(threads)

  print("recommendations: " + str(len(recommendations)))
  pprint.pprint(recommendations)
