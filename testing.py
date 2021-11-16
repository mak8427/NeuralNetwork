import pandas as pd
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

df=pd.read_csv('data.csv')
df1=pd.read_csv('data_decay.csv')

print(df)
plt.plot(df['accuracy'].head(10000))
plt.show()
plt.plot(df1['accuracy'])

plt.show()

X, y = spiral_data( samples = 100 , classes = 3 )
print(X)