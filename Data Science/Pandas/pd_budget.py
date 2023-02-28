# Pandas example: read csv and do some statistical analysis and plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. read csv with pandas, set the first column as the index
data = pd.read_csv('budget.csv', index_col=0)
print(data)
print('--------------------------------')

# 2. plot for statistical average budget of each year
# temp = np.array(data['2018'])
# print(type(temp))
avg_budget = [np.mean(data['2018']),np.mean(data['2019']),np.mean(data['2020'])]
print(avg_budget)
fig, ax = plt.subplots()
years = ['2018', '2019', '2020']
bar_colors = ['tab:red', 'tab:blue', 'tab:pink']
ax.bar(years, avg_budget, color=bar_colors)
ax.set_ylabel('Average Budget')
plt.show()
print('--------------------------------')

# 3. statistical analysis for one person, search by label using index
person = data.loc['David']
print(np.mean(person))
print(np.std(person))