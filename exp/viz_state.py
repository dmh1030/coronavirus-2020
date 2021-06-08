"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import censusdata

plt.style.use('fivethirtyeight')


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
NUM_COLORS = 0
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_state(df)
    cases = cases.sum(axis=0)

    if cases.sum() > MIN_CASES:
        NUM_COLORS += 1

colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []

for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_state(df)
    cases = cases.sum(axis=0)

    if cases.sum() > MIN_CASES:
        i = len(legend)
        lines = ax.plot(cases, label=labels[0])
        handles.append(lines[0])
        lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])
        lines[0].set_color(colors[i])
        legend.append(labels[0])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

ax.set_yscale('log')
ax.legend(handles, legend, bbox_to_anchor=(-0.05, 1.05, 1.0, .100), loc=3, ncol=4)
plt.tight_layout()
plt.title("Cases per State")
plt.savefig('results/cases_by_state.png')



#by population
pop_data = censusdata.download('acs5', 2015, censusdata.censusgeo([('state', '*')]), ['B01003_001E'])
pop_data = pd.DataFrame(pop_data)
pop_data.columns = ['Population']

new_indices = []
state_names = []

for index in pop_data.index.tolist():
    new_index = index.geo[0][1]
    new_indices.append(new_index)
    state_names.append(index.name)
pop_data.index = new_indices
pop_data.index = state_names
pop_data.sort_index()

print(pop_data.loc['Texas', 'Population'] + pop_data.loc['California', 'Population'])



fig2 = plt.figure(figsize=(12, 12))
ax2 = fig2.add_subplot(111)
cm = plt.get_cmap('jet')

#colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend2 = []
handles2 = []
for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_state(df)
    cases = cases.sum(axis=0)
    if val not in pop_data.index:
        print(val)
        continue
    state_pop = pop_data.loc[val, 'Population']
    cases = (cases / state_pop)* 1000

    if cases.sum() > MIN_CASES:
        i = len(legend2)
        lines2 = ax2.plot(cases, label=labels[0])
        handles2.append(lines2[0])
        lines2[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])
        lines2[0].set_color(colors[i])
        legend2.append(labels[0])


ax2.set_ylabel('Confirmed Cases per 1000 people')
ax2.set_xlabel("Time (days since Jan 22, 2020)")

#ax2.set_yscale('log')
ax2.legend(handles, legend, bbox_to_anchor=(-0.05, 1.05, 1.0, .100), loc=3, ncol=4)
plt.tight_layout()
plt.title("Cases per State per 1000 people")
plt.savefig('results/cases_by_state_pc.png')

