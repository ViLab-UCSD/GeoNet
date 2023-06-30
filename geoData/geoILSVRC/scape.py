#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import flickrapi
from tqdm.notebook import tqdm

import string, urllib

import warnings
warnings.filterwarnings('ignore')

import flickrapi
from datetime import datetime

# Set your Flickr API key and secret
api_key = u'69e05482ffba962ac43605e50c6dc800'
api_secret = u'6bcdd054a5518a08'

# Set the date and time after which photos should be uploaded
min_upload_date = datetime(2014, 1, 1, 0, 0, 0).strftime('%s')

# Set bbox coordinates for Asia
bbox = '31.289063,-17.140790,164.179,57.12'  # longitude, latitude format

# Initialize the Flickr API client
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

mapping = json.load(open("/home/tarun/GeoDA_Project/geoData/geoILSVRC/geoimnet_classMapping.txt"))

for name, syns in mapping.items():
    added_syns = []
    for s in syns:
        added_syns.extend([s.replace(" ","_"), s.replace(" ","-"), s.replace(" ","")])
    
    mapping[name] = list(set(syns + added_syns))

all_photos = dict()

def counts_per_user(photo_ids, MAX_COUNT=20):
    
    # Retrieve the photo IDs and count images per user
    pruned_photo_ids = []
    image_count_per_user = {}
    visited_ids = set()
    
    for photo in photo_ids:
        owner = photo['owner']
        if owner in image_count_per_user:
            if image_count_per_user[owner] >= MAX_COUNT:
                continue
            else:
                image_count_per_user[owner] += 1
        else:
            image_count_per_user[owner] = 1
        
        curr_photo_id = photo["id"]
        if curr_photo_id not in visited_ids:
            visited_ids.add(curr_photo_id)
            pruned_photo_ids.append(photo)
            
    return pruned_photo_ids

for classname in tqdm(mapping.keys()):
    
    tag_list = mapping[classname]
    print("Downloading for {}".format(classname))
    
    photo_id = []
    for tag in tag_list:
        
        counter = 0
        
        while len(photo_id) < 600 and counter < 150:
            
            try:
                photos = flickr.photos.search(tags=tag, min_upload_date=min_upload_date, per_page=250, page=counter, has_geo=1, bbox=bbox, content_type=1, extras = ["date_taken", "geo", "license", "tags", "owner_name", "url_c"], license='1,2,3,4,5,6,7')
            except:
                continue
            
            counter += 1
            print("Counter id: {}".format(counter), end="\r")
            photo_id.extend(photos['photos']['photo'])
            photo_id = counts_per_user(photo_id)
                
            
        
    all_photos[classname] = photo_id
    print("Counter id: {}, Photos Collected: {}".format(counter, len(all_photos[classname])), end="\r")
    # print()

with open("/home/tarun/GeoDA_Project/geoData/geoILSVRC/challenge_test.json", "w") as fh:
    json.dump( {"challenge_imnet":all_photos}, fh)

