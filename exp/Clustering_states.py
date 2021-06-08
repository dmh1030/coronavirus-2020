import sys
sys.path.insert(0, '..')
from sklearn.cluster import KMeans
from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

import pandas as pd
import censusdata

pop_data = censusdata.download('acs5', 2015, censusdata.censusgeo([('state', '*')]), ['B01003_001E'])
df = pd.DataFrame(pop_data)
df.columns = ['Population']

new_indices = []
state_names = []

for index in df.index.tolist():
    new_index = index.geo[0][1]
    new_indices.append(new_index)
    state_names.append(index.name)
df.index = new_indices
df.index = state_names
df.sort_index()
df_states = df.index

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'

# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
states = np.unique(confirmed["Province_State"])
print("States", states.shape)
states = states.tolist()

features = np.zeros((len(df.index), 498))
targets = []

i = 0
for state in states:
    df2 = data.filter_by_attribute(
        confirmed, "Province_State", state)
    cases, labels = data.get_cases_chronologically_state(df2)
    #cases = cases / 1000000
    if cases.shape[0] > 1:
        cases = cases.sum(axis=0)
    if state not in df_states:
        print(state)
        continue
    state_pop = df.loc[state, 'Population']
    features[i] = (cases / state_pop) * 1000
    targets.append(state)
    i += 1

model = KMeans(5)
model.fit(features)
predicted = model.predict(features)

#print("Predictions: ", predicted)

group0 = np.zeros(features.shape[1])
group1 = np.zeros(features.shape[1])
group2 = np.zeros(features.shape[1])
group3 = np.zeros(features.shape[1])
group4 = np.zeros(features.shape[1])

c0 = []
c1 = []
c2 = []
c3 = []
c4 = []

count = np.zeros((5,))

for i in range(features.shape[0]):
    if predicted[i] == 0:
        count[0] += 1
        group0 = np.add(group0, features[i])
        c0.append(targets[i])
    elif predicted[i] == 1:
        count[1] += 1
        group1 = np.add(group1, features[i])
        c1.append(targets[i])
    elif predicted[i] == 2:
        count[2] += 1
        group2 = np.add(group2, features[i])
        c2.append(targets[i])
    elif predicted[i] == 3:
        count[3] += 1
        group3 = np.add(group3, features[i])
        c3.append(targets[i])
    elif predicted[i] == 4:
        count[4] += 1
        group4 = np.add(group4, features[i])
        c4.append(targets[i])
    else:
        print("somethings wacked")

print("Group Counts: ", count)

group0 = group0 / count[0]
group1 = group1 / count[1]
group2 = group2 / count[2]
group3 = group3 / count[3]
group4 = group4 / count[4]

''''''
print('\nGroup 0: ', c0)
print('\nGroup 1: ', c1)
print('\nGroup 2: ', c2)
print('\nGroup 3: ', c3)
print('\nGroup 4: ', c4)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
zero, = plt.plot(range(498), group0, color='red')
one, = plt.plot(range(498), group1, color='orange')
two, = plt.plot(range(498), group2, color='green')
three, = plt.plot(range(498), group3, color='blue')
four, = plt.plot(range(498), group4, color='purple')
#ax.set_yscale('log')
plt.legend([zero, one, two, three, four], ['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4'])
plt.xlabel("Time (days since Jan 22, 2020)")
plt.ylabel('Confirmed cases per 1000 People')
plt.title('States Clustered By Cases')
plt.savefig('results/us_clustered.png')






