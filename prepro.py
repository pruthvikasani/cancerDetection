import pandas as pd
import numpy as np
import matplotlib as plt
df = pd.read_csv("dataset.csv")

df.head(10)

df.describe()

df['Smoking'].hist(bins=30)

df.boxplot(column='Smoking')

df.boxplot(column='Smoking', by = 'Alcohol use')

df.apply(lambda x: sum(x.isnull()),axis=0)