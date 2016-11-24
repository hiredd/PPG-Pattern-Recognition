import csv
import sys
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
from os import listdir
from datetime import datetime, timedelta
import os.path

files_path = "/home/gal/Downloads/thesis_data"

data = []
data_file_save = "data/data%d.csv"
data_point_names = "timestamp,device,ppg,accx,accy,accz".split(",")

sample_freq = 80
time_delta = timedelta(milliseconds = (10**3)/sample_freq)

for file_id, file_name in enumerate(listdir(files_path)):

    filename_components = file_name.split("_")
    file_device_id = filename_components[2].replace(".txt","")
    print(file_device_id)

    timestamp = datetime.strptime("2016:" + ":".join(filename_components[:2]) + ":9", "%Y:%m:%d:%H")

    with open(files_path + "/" + file_name) as f:
        content = f.readlines()

        li = 0

        lenC = len(content)

        while li < lenC:

            # add an if for "!" - new write to file on spansion memory
            if(content[li].strip() == "!"):
                data.append(["0" for _ in np.zeros(6)])
                li += 1

            datapoint = {"P": -1, "A": -1}
            setP, setA = False, False
            should_skip = False
            while li < lenC and (setP == False or setA == False):

                if(content[li].strip() == "!"):
                    data.append(["0" for _ in np.zeros(6)])
                    should_skip = True
                    break

                if li < lenC and len(content[li]) > 2:
                    if setP == False and content[li][:2] == "P:":
                        datapoint["P"] = float(content[li][2:].strip())
                        setP = True
                    elif setA == False and content[li][:2] == "A:" and content[li][2:].count(":") == 2:
                        datapoint["A"] = np.asfarray(content[li][2:].strip().split(":"))
                        setA = True
                li += 1

            if li == lenC:
                break

            if should_skip:
                continue

            data.append([str(timestamp), 
                         str(file_device_id), 
                         str(datapoint["P"]), 
                         str(datapoint["A"][0]), 
                         str(datapoint["A"][1]), 
                         str(datapoint["A"][2])])

            timestamp = timestamp + time_delta

    df = pd.DataFrame(data)
    #if not os.path.isfile(data_file_save % file_id):
    df.to_csv(data_file_save % file_id, sep=',', encoding='utf-8', header=data_point_names)
    # else:
    #     with open(data_file_save, 'a') as f:
    #         print("writing")
    #         df.to_csv(f, sep=',', encoding='utf-8', header=False)
    data = []

#columns=["id","timestamp","device","PPG","acc_x","acc_y","acc_z"]