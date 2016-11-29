import os
import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import col, collect_list, udf
from pyspark.sql import DataFrame
from datetime import datetime 
import pyspark.sql.types as ptypes
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import welch
from classes.Signal import Signal
from classes.HRClassifier import HRClassifier

def __(x):
	print("==========" + str(x) + "==========")

#conf = SparkConf().setMaster("local").setAppName("HR Classifier")
#sc = SparkContext(conf = conf)

#sqlContext = SQLContext(sc)

classifier = HRClassifier()

def dothis(x):
	return x

classifier.load_data1()
classifier.train1()
quit()












for file_id in range(num_data_files):
	classifier.load_data(sqlContext, sc, 1)

	quit()

	d = classifier.HR_ranges.select(["start", "end"]).where(col("file_id") == 1)
		#lambda x: classifier.data.select("*").where(col("record_id") >= x.start).where(col("record_id") <= x.end))

	__(d.first())
	quit()

	# select rows in the ranges specified in classifier
	data = classifier.data.select("*").where(col("record_id") >= valid_range_start).where(col("record_id") <= valid_range_end)
	rows.collect()
	#turn into signals, take logPSDs, feed to classifier.train


	i = 1
	fig = plt.figure(file_id+1)

	rows = 1+len(HRranges[identifier])//3
	for valid_range in HRranges[identifier]:
		valid_range_start,valid_range_end = valid_range

		data = df.select("*")\
				 .where(col("record_id") >= valid_range_start-3000)\
				 .where(col("record_id") <= valid_range_end+3000)
		data_rdd = data.rdd
		ppg_segment = data_rdd.map(lambda x: x.ppg).collect()
		timestamps = data_rdd.map(lambda x: x.timestamp).collect()

		signal = Signal(ppg_segment, timestamps)
		signal.correct_saturation()
		signal.remove_outliers()

		classifier.extract_features(signal, is_valid_HR = True)

		plt.subplot(rows,3,i)
		plt.title("file: %s, start: %d, end: %d" % (identifier, valid_range_start, valid_range_end))
		plt.plot(signal.content)
		i += 1

	if file_id == 4:
		break

#classifier.train()

plt.show()

sc.stop()