import os
import sys
from datetime import datetime 
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import welch
from classes.Signal import Signal
from classes.HRClassifier import HRClassifier

classifier = HRClassifier()
classifier.train(True)
classifier.validate(20)
