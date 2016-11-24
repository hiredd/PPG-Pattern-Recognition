import csv
import sys
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
from os import listdir
from datetime import datetime, timedelta


# BIG PROBLEM: times are all off in the files, not much can be done about that


files_path = "/home/gal/Downloads/thesis_data"

data = []
written = False

time_diff = timedelta(milliseconds = (10**3)/80)

for file_name in listdir(files_path):
    print(file_name)

    filename_components = file_name.split("_")
    file_device_id = filename_components[2].replace(".txt","")
    begin_time = datetime("2016" + ":".join(filename_components[:2]) + ":9", "%Y:%m:%d:%H")

    with open(files_path + "/" + file_name) as f:
        content = f.readlines()

        li = 0
        while li < len(content):

            ts = content[li][2:].strip().split(":")
            time_start_s = content[li][2:]
            time_start = datetime.strptime(":".join(ts[:-1]), "%Y:%m:%d:%H:%M:%S") + timedelta(milliseconds = int(ts[-1]))

            # Accumulate PPG, Accelerometer readings until the following timestamp
            PPG_ACC = []
            li += 1
            while li < len(content) and content[li][:2] != "T:":
                datapoint = {"P": -1, "A": -1}
                setP, setA = False, False
                while li < len(content) and setP == False or setA == False:
                    if li < len(content) and len(content[li]) > 2:
                        if setP == False and content[li][:2] == "P:":
                            datapoint["P"] = float(content[li][2:].strip())
                            setP = True
                        elif setA == False and content[li][:2] == "A:" and content[li][2:].count(":") == 2:
                            datapoint["A"] = np.asfarray(content[li][2:].strip().split(":"))
                            setA = True
                    li += 1

                PPG_ACC.append(datapoint)
                li += 1

            if li == len(content):
                break

            time_end_s = content[li][2:].strip()
            time_end = datetime.strptime(content[li][2:].strip(), "%Y:%m:%d:%H:%M:%S:%f")

            if time_start > time_end:
                print("stopped")
                print(time_start, time_end)
                print(time_start_s, time_end_s)
                break
            
            # Calculate the timestamp for each PPG, Acc reading pair
            delta = (time_end-time_start)/len(PPG_ACC)

            last_timestamp = time_start
            for reading in PPG_ACC:
                data.append([str(last_timestamp), 
                             str(file_device_id), 
                             str(reading["P"]), 
                             str(reading["A"][0]), 
                             str(reading["A"][1]), 
                             str(reading["A"][2])])

                last_timestamp += delta

    if len(data) > 0:
        df = pd.DataFrame(data)
        if written == False:
            df.to_csv("data.csv", sep=',', encoding='utf-8')
        else:
            with open('data.csv', 'a') as f:
                df.to_csv(f, sep=',', encoding='utf-8', header=False)
            written = True
    data = []

#columns=["id","timestamp","device","PPG","acc_x","acc_y","acc_z"]