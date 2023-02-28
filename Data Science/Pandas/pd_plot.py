# Pandas example: Plot using Pandas example
# modified from: https://github.com/MorvanZhou/tutorials/blob/master/numpy%26pandas/18_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Series
data = pd.Series(np.random.randn(500), index=np.arange(500))
data = data.cumsum()    # accumulated sum
data.plot()
plt.show()

# 2. DataFrame
data = pd.DataFrame(np.random.randn(500, 5), index=np.arange(500), columns=list("ABCDE"))
data = data.cumsum()
data.plot()
plt.show()

# 3. scatter plot
ax = data.plot.scatter(x='A', y='B', color='Blue', label="Positive")
data.plot.scatter(x='A', y='C', color='Green', label='Negative', ax=ax)
data.plot.scatter(x='A', y='D', color='Red', label='Invalid', ax=ax)
plt.show()