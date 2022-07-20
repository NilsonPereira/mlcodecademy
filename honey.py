import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")

tp = df.groupby('year').mean()

print(tp.head())

print("OK")