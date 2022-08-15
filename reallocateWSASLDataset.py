import json
import os
from os.path import exists

json_file_path = 'D:\capstone\capstone\WSASLinfo\WLASL_v0.3.json'

with open(json_file_path, 'r') as json_file:
    wlasl_data = json.load(json_file)
    for word in wlasl_data:
        gloss = word["gloss"]
        for instance in word["instances"]:
            old_path = "D:/capstone/capstone/videos_original/all/" + instance["video_id"] + ".mp4"
            new_path = "D:/capstone/capstone/videos_original/classes/" + gloss + "/" + instance["video_id"] + ".mp4"
            if exists(old_path):
                os.rename(old_path, new_path)
                print("moving " + old_path + " to " + new_path)