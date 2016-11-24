from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from datetime import datetime 
import pyspark.sql.types as ptypes
import matplotlib.pyplot as plt
import numpy as np
import pandas
from classes.Signal import Signal

HRranges = {
    '6_2_1': [(61911,64962),(65446,67882),(80208,85046),(88139,90207),(223169,223899),(232908,234236),(265489,267293),(353947,354697),(366813,371476),(411791,415532),(448336,449363)],
    '6_2_2': [(203981,208131),(215986,217841),(221827,222478),(437989,439886),(458251,459258),(493759,494177)],
    '6_2_3': [(221380,222809)],
    '6_1_1': [(391997,395445),(396675,400513),(411186,415921),(425726,426439),(430232,432188),(439971,441243),(491068,492569)],
    '6_1_2': [(143511,143975),(392455,392983),(398491,399516),(400320,401172),(402684,403640),(403854,407949),(408882,412079),(415804,416261),(436978,437527),(443985,444455),(445525,446376)],
    '6_1_3': [(113552,115794)],
    '6_1_4': [(55912,56500),(56991,58063),(61791,62657),(67132,67983),(68948,70386),(70585,71657),(73099,74212),(79848,80867),(86573,89705),(94715,96642),(96735,102051),(105413,106107),(108072,108808),(109292,118669),(126105,126586),(131356,133058),(133747,134458),(135640,136572),(137681,139264),(140290,143900),(146079,147153),(148802,152134),(154275,155777),(246174,263619),(278957,285305),(286130,289059),(295653,302292),(307965,310724),(361960,363442),(366771,367844),(379721,381872),(384094,397889),(401916,402939),(404748,408671),(409326,415055),(416749,425676),(427470,442650),(455548,456752),(466106,467333),(468511,469468),(469943,470837),(494808,497432),(580152,589637),(602129,602786),(603287,604102),(622554,623635),(624297,626239),(631506,633657),(652580,653997),(666283,667131),(667721,668139)]
}

# SparkSession available as spark

def __(x):
	print("==========" + str(x) + "==========")

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)

customSchema =  ptypes.StructType([
    ptypes.StructField("_c0",  ptypes.IntegerType(), True),
    ptypes.StructField("timestamp",  ptypes.StringType(), True),
    ptypes.StructField("device",  ptypes.IntegerType(), True),
    ptypes.StructField("ppg",  ptypes.DoubleType(), True),
    ptypes.StructField("accx",  ptypes.DoubleType(), True),
    ptypes.StructField("accy",  ptypes.DoubleType(), True),
    ptypes.StructField("accz",  ptypes.DoubleType(), True)])

sqlContext = SQLContext(sc)

for file_id in range(20):
	df = sqlContext.read.csv("data/data%d.csv" % file_id, header = True, schema=customSchema)

	row = df.select(["timestamp", "device"]).take(5)[-1]
	device_id = row.device
	date = datetime.strptime(row.timestamp[:10], '%Y-%m-%d')
	identifier = "%d_%d_%d" % (date.month, date.day, device_id)

	if identifier not in HRranges:
		continue

	lines = df.select("ppg").rdd.map(lambda p: p.ppg).collect()

	#lines.persist()
	i = 1
	fig = plt.figure(file_id+1)

	rows = 1+len(HRranges[identifier])//3
	for valid_range in HRranges[identifier]:
		valid_range_start,valid_range_end = valid_range

		s = Signal(lines[valid_range_start-3000:valid_range_end+3000])
		s.correctSaturation()

		plt.subplot(rows,3,i)
		plt.title("file: %s, start: %d, end: %d" % (identifier, valid_range_start, valid_range_end))
		plt.plot(s.content)
		i += 1

plt.show()

sc.stop()