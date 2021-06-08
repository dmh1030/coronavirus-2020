import sys
sys.path.insert(0, '..')

from utils import data
import os
import numpy as np
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

red = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'Florida', 'Georgia', 'Idaho', 'Indiana', 'Iowa', 'Mississippi',
       'Massachusetts', 'Maryland', 'Missouri', 'Montana', 'Nebraska', 'New Hampshire', 'North Dakota', 'Ohio',
       'Oklahoma', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'West Virginia', 'Wyoming']

blue = ['California', 'Colorado', 'Connecticut', 'Delaware', 'Hawaii', 'Illinois', 'Kansas',
        'Kentucky', 'Louisiana', 'Maine', 'Michigan', 'Minnesota', 'Nevada', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'Oregon', 'Pennsylvania', 'Rhode Island',  'Virginia', 'Washington',  'Wisconsin']

red_pop = 0
blue_pop = 0

for state in red:
    red_pop += df.loc[state, 'Population']
for state in blue:
    blue_pop += df.loc[state, 'Population']

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
NUM_COLORS = 0
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

states = np.unique(confirmed["Province_State"])
red_cases = np.ones((498,))
blue_cases = np.ones((498,))

for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_state(df)
    cases = cases
    cases = cases.sum(axis=0)
    if val in red:
        red_cases = np.add(red_cases, cases)
    if val in blue:
        blue_cases = np.add(blue_cases, cases)

red_cases = (red_cases / red_pop) * 1000
blue_cases = (blue_cases / blue_pop) * 1000

red, = ax.plot(red_cases, label='Red', color='red')
blue, = ax.plot(blue_cases, label='Blue', color='blue')
ax.legend([red, blue], ["Republican", 'Democrat'], loc=3, ncol=4)
ax.set_ylabel('Cases per 1000 People')
ax.set_xlabel("Time (days since Jan 22, 2020)")
plt.title("Cases Divided by Blue or Red Governors")
plt.tight_layout()
plt.savefig('results/cases_by_gov.png')


