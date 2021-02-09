import json
import pandas as pd
from datetime import datetime
import os
import jsonlines
import ember


def filter_and_recreate_json_by_year(dataset,year,new_folder_name, month=None):
    
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

    for folder in range(7):
        if folder == 6:
            json_file_name = 'test_features.jsonl'
            json_file_path = '{}/{}'.format(dataset,json_file_name)
        else:
            json_file_name = 'train_features_{}.jsonl'.format(folder)
            json_file_path = '{}/train_features_{}.jsonl'.format(dataset,folder)
        
        cur_folder = '{}/{}'.format(new_folder_name,json_file_name)
        
        with jsonlines.open(cur_folder, 'w') as writer:
            with open(json_file_path, 'r') as json_file:
                json_list = list(json_file)
                for json_str in json_list:
                    json_content = json.loads(json_str)
                    appreaed_time = json_content['appeared'].split('-')
                    if (appreaed_time[0]==year and (appreaed_time[1]==month or appreaed_time[1] == None)):
                        del json_content['appeared']
                        writer.write(json_content)

ls = ['01','02','03','04','05','06','07','08','09','10']

for i in ls:
    filter_and_recreate_json_by_year('ember_2017_2','2017','ember_2017_2_{}'.format(i), i)

    try:
        ember.create_vectorized_features("ember_2017_2_{}".format(i))
    except:
        ember.create_metadata("ember_2017_2_{}".format(i))
    else:
        ember.create_metadata("ember_2017_2_{}".format(i))

    











# filter_and_recreate_json_by_year('ember_2017_2','2017','new_ember_2017_2')

# filter_and_recreate_json_by_year('ember2018','2018','new_ember2018')

# ember.create_vectorized_features("new_ember_2017_2")
# ember.create_metadata("new_ember_2017_2")

# ember.create_vectorized_features("new_ember2018")
# ember.create_metadata("new_ember2018")



