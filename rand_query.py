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
def randomQuery:
  test_query = songs.aggregate(
   [ { '$sample': { "size": 100 } } ]
)
