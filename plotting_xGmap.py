import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./All shots from 16-17/E0/shots with proba.csv", index_col=0)

data = data[data["Match No"] == 0]

for index, row in data.iterrows():
    if row['Team'] == 'burnley':
        data.set_value(index, 'x', 480 - row['x'])

data['y'] = -1 * data['y']

plt.scatter(data['x'], data['y'], s=data['Proba_exp']*500, c=data['Colour'])
plt.ylim(-366/2, 366/2)
plt.xlim(0, 480)
plt.show()

