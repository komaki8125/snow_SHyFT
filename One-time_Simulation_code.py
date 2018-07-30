'''
This code does one-time simulation and gives out some more data about the model which are commented in following
	Double check the main model data
	Make Precipitation and Temperature graphs
	Make a Data-Frame and put discharge of all sub-catchments into a CSV file
	Make a Data-Frame and put the distributed p, T, Geo and etc. into separated CSV file
	Generate SCA, SWE and outflow of all cells
	Generate all SCA & SWE images
	Discharge graphs of all targets
	make precipitation graph

'''
#  Importing the third-party python modules

from netCDF4 import Dataset
import os
from os import path
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
#  Recored the starting time

t1 = time.time()


#  Define SHyFT data path
#  Adding 'r' to avoid change slash or doubl backslash

shyft_data_path = path.abspath(r"C:\shyft_workspace\shyft-data")
if path.exists(shyft_data_path) and 'SHYFT_DATA' not in os.environ:
    os.environ['SHYFT_DATA']=shyft_data_path


#  Importing the shyft modules

import shyft
from shyft import api
from shyft.repository.default_state_repository import DefaultStateRepository
from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
from shyft.orchestration.simulators.config_simulator import ConfigSimulator


#  Set up YAML files to configure simulation 

config_file_path = r'D:\Dropbox\Thesis\SHyFT\neanidelva_simulation.yaml'
cfg = YAMLSimConfig(config_file_path, "neanidelva")


#  Config the simulator

simulator = ConfigSimulator(cfg)
region_model = simulator.region_model


#  Double check the main information of the model

print('Number of steps is','\t\t\t', cfg.number_of_steps,'\n')
print('Start datetime is','\t\t\t', cfg.start_datetime,'\n')
print('Number of seconds of each step is','\t', cfg.run_time_step,'\n')
print('Name and method of model is','\t\t', cfg.region_model_id,'\n')
print('Number of total cells are','\t\t', simulator.region_model.size(),'\n')
print('catchment_ids are:\n\n',simulator.region_model.catchment_ids,'\n')


#  Run the simulation

simulator.run()


#  Make Precipitation and Temperature graph for a catchment or a cell in a period

while True:
    question = input("make a P & T graph, for a catchment or a cell?")
    if question == 'catchment' or question == 'cell' or question == 'stop':
        break
if question == 'catchment':

    print (region_model.catchment_ids)
    cid = int(input("Please enter catchment ID"))
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))

    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
    
    pre_cell = region_model.statistics.precipitation([cid]).values
    temp_cell = region_model.statistics.temperature([cid]).values
    
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()    
    ax1.plot(ts_timestamps,pre_cell[start_day:n_day+start_day], c='black', lw=2, label='precipitation')
    ax2.plot(ts_timestamps, temp_cell[start_day:n_day+start_day], c='purple', lw=2, label='Temperature')
    ax1.set_ylabel('daily precip [mm/h]')
    ax2.set_ylabel('temp [$°$ C]')
    ax1.set_xlabel('date')
    # loc = 1(right-top) 2(left-top) 3(bottom-left) 4(bottom-right)
    ax1.legend(loc=2); ax2.legend(loc=1)
    plt.show()
    print("precipitation = ", pre_cell[start_day:n_day+start_day])
    print("Temperature = ", temp_cell[start_day:n_day+start_day])
    

elif question == 'cell':
    
    print (f"Total number of cells are {simulator.region_model.size()}, enter from 0 to {simulator.region_model.size()-1}")
    cell_num = int(input("Enter the cell id"))
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))    

    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]    
    
    pre_cell = region_model.cells[cell_num].env_ts.precipitation.values
    temp_cell = region_model.cells[cell_num].env_ts.temperature.values    
    
    
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()    
    ax1.plot(ts_timestamps,pre_cell[start_day:n_day+start_day], c='cornflowerblue', lw=2, label='precipitation')
    ax2.plot(ts_timestamps, temp_cell[start_day:n_day+start_day], c='orange', lw=2, label='Temperature')
    ax1.set_ylabel('daily precip [mm/h]')
    ax2.set_ylabel('temp [$°$ C]')
    ax1.set_xlabel('date')
    # loc = 1(right-top) 2(left-top) 3(bottom-left) 4(bottom-right)
    ax1.legend(loc=2); ax2.legend(loc=1)
    plt.show()
    print("precipitation = ", pre_cell[start_day:n_day+start_day])
    print("Temperature = ", temp_cell[start_day:n_day+start_day])
else:
    pass


#  Make a pandas DataFrame and put discharge of all subcatchments and save into a CSV file

discharge_subcatch_pd = pd.DataFrame()
for cid in region_model.catchment_ids:
    discharge_subcatch_pd[cid] = region_model.statistics.discharge([int(cid)]).values 

ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]    
discharge_subcatch_pd.index = ts_timestamps
discharge_subcatch_pd.to_csv('discharge_subcatch_pd.csv')


#  Access to the whole dataframe

discharge_subcatch_pd.loc[:][:]


#  Access to the specific catchment and time

discharge_subcatch_pd.loc['2014-03-18'][1996]


#  Access to discharge of all catchments in specific date

discharge_subcatch_pd.loc['2014-03-18'][:]


#  Access to discharge of specific catchment in whole period

discharge_subcatch_pd.loc[:][1996]


#  Make a pandas DataFrame and put the distributed precipitation,radiation,relative_humidity,temperature,wind_speed and discharge of all cells and save into CSV files

precipitation_pd = pd.DataFrame()
radiation_pd = pd.DataFrame()
rel_hum_pd = pd.DataFrame()
temperature_pd = pd.DataFrame()
wind_speed_pd = pd.DataFrame()
disch_cell_pd = pd.DataFrame()

for num in range(region_model.size()):
    precipitation_pd[num] = region_model.cells[num].env_ts.precipitation.values
    radiation_pd[num] = region_model.cells[num].env_ts.radiation.values
    rel_hum_pd[num] = region_model.cells[num].env_ts.rel_hum.values
    temperature_pd[num] = region_model.cells[num].env_ts.temperature.values
    wind_speed_pd[num] = region_model.cells[num].env_ts.wind_speed.values
    disch_cell_pd[num] = region_model.cells[num].rc.avg_discharge.values
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]    

precipitation_pd.index = ts_timestamps
radiation_pd.index = ts_timestamps
rel_hum_pd.index = ts_timestamps
temperature_pd.index = ts_timestamps
wind_speed_pd.index = ts_timestamps
disch_cell_pd.index = ts_timestamps

precipitation_pd.to_csv('precipitation_pd.csv')
radiation_pd.to_csv('radiation_pd.csv')
rel_hum_pd.to_csv('rel_hum_pd.csv')
temperature_pd.to_csv('temperature_pd.csv')
wind_speed_pd.to_csv('wind_speed_pd.csv')
disch_cell_pd.to_csv('disch_cell_pd.csv')


#  Make discharge graph for a catchment or a cell in a period

while True:
    question = input("make a graph for a catchment or a cell?")
    if question == 'catchment' or question == 'cell' or question == 'stop':
        break
if question == 'catchment':

    print (region_model.catchment_ids)
    cid = input("Please enter catchment ID")
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))

    fig, ax = plt.subplots(figsize=(10,8))
    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
    data = region_model.statistics.discharge([int(cid)]).values
    ax.plot(ts_timestamps,data[start_day:n_day+start_day], label = "{}".format(cid))
    
    fig.autofmt_xdate()
    ax.legend(title="Catch. ID")
    ax.set_ylabel("discharge [m3 s-1]")
    plt.show()
    print(data[start_day:n_day+start_day])
elif question == 'cell':
    
    print (f"Total number of cells are {simulator.region_model.size()}, enter from 0 to {simulator.region_model.size()-1}")
    cell_num = int(input("Enter the cell id"))
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))    

    fig, ax = plt.subplots(figsize=(10,8))
    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

    data = region_model.cells[cell_num].rc.avg_discharge.values
    ax.plot(ts_timestamps,data[start_day:n_day+start_day], label = f"Cell {cell_num}")
    
    fig.autofmt_xdate()       
    ax.legend(title="Cell ID")
    ax.set_ylabel("discharge [m3 s-1]")
    plt.show()
    print(data[start_day:n_day+start_day])
else:
    pass


#  Access to the x, y ,z, area and catch_ids of all cells

cells = region_model.get_cells()

x = np.array([cell.geo.mid_point().x for cell in cells])
y = np.array([cell.geo.mid_point().y for cell in cells])
z = np.array([cell.geo.mid_point().z for cell in cells])
area = np.array([cell.geo.area() for cell in cells])
catch_ids = np.array([cell.geo.catchment_id() for cell in cells])


#  Make a panadas DataFrame for Geo. data save into a CSV file

geo_pd = pd.DataFrame()
geo_pd['x'] = x
geo_pd['y'] = y
geo_pd['z'] = z
geo_pd['catch_ids'] = catch_ids
geo_pd['area'] = area
geo_pd.to_csv('geo_pd.csv')


#  Do some calculation on Geo. data

np_z = np.array(geo_pd[:]['z'])
print(np_z.size)
print(np_z.mean())
print(np_z.max())
print(np_z.min())
print(np_z.std())


#   Access directly to the catchment_ids

catchment_ids = region_model.catchment_ids


#   Make a dictionary an enumarate them form zero to twenty-six

cid_z_map = dict([(catchment_ids[i],i) for i in range(len(catchment_ids))])
print(cid_z_map)


#   Then create an array the same length as our 'x' and 'y', which holds the integer reflecting values with the cid_z_map dictionary for each single cells

catch_ids = np.array([cid_z_map[cell.geo.catchment_id()] for cell in cells])


#   Illustrate the catchment

fig, ax = plt.subplots(figsize=(15,5))
cm = plt.cm.get_cmap(color[73])# color[0 to 75]
plot = ax.scatter(x, y, c=catch_ids, marker='s', s=7, lw=4, cmap=cm)
# plot = ax.scatter(x, y, c=z, marker='o', s=9, lw=5, cmap=cm)
plt.colorbar(plot).set_label('Numerate the sub-catchment IDs')
# plt.legend(title="sub-catchments", fontsize = 16, loc = 1)
plt.show()


#   Gamma-snow response
#   Set a date: year, month, day, (hour of day if hourly time step). The oslo calendar(incl dst) converts calendar coordinates Y,M,D.. to its utc-time. 1400104800 (seconds passed from 1970,1,1,1,0,0). It needs to get the index of the time_axis for the time

oslo = api.Calendar('Europe/Oslo') # Europe/Berlin
time_x = oslo.time(2016,3,1)


#   Index of time x on time-axis

try:
    idx = simulator.region_model.time_axis.index_of(time_x)
except:
    print("Date out of range, setting index to 0")
    idx = 0


#   Snow Cover Area
#   In the mentioned day idx = (2016,3,1) for all all catchments ([])

sca = simulator.region_model.gamma_snow_response.sca([],idx)
# sca = simulator.region_model.hbv_snow_state.sca([],idx)
# sca = simulator.region_model.skaugen_snow_state.sca([],idx)

#   Snow Water Equivalent (mm)
#   In the mentioned day idx = (2016,3,1) for all catchments ([])

swe = simulator.region_model.gamma_snow_response.swe([],idx)
# swe = simulator.region_model.hbv_snow_state.swe([],idx)
# swe = simulator.region_model.skaugen_snow_state.swe([],idx)


#   The average of swe in the selected catchment, one value (mm)

swev = simulator.region_model.gamma_snow_response.swe_value([],idx)
# swev = simulator.region_model.hbv_snow_state.swe_value([],idx)
# swev = simulator.region_model.skaugen_snow_state.swe_value([],idx)

swe_np = np.array(swe)
area_np = np.array(area)
sum_np = swe_np * area_np
swe_average = sum_np.sum()/area_np.sum()
print(swe_average)
print(swev)
print(round(swe_average,3) == round(swev,3))


#   Do some calculation with numpy help

print(swe_np.mean())
print(swe_np.std())
print(swe_np.sum())
print(swe_np.max())
print(swe_np.min())


# The number of cells with more 250 mm Snow Water equivalent

swe_np[swe_np > 250].size


#   Snow outflow

sout = simulator.region_model.gamma_snow_response.outflow([],idx)
# sout = simulator.region_model.hbv_snow_response.outflow([],idx)
# sout = simulator.region_model.skaugen_snow_response.outflow([],idx)

sout_np = np.array(sout)
print(sout_np)
print(sout_np.sum())
print(sout_np.max())
print(sout_np.min())


#   Simple scatter plots for SCA , SWE , Outflow

fig, ax = plt.subplots(figsize=(15,5))
cm = plt.cm.get_cmap(color[2])# color[0 to 75]
plot = ax.scatter(x, y, c=sca, vmin=0, vmax=1,marker='s', s=40, lw=0, cmap=cm)
plt.colorbar(plot)
plt.title('Snow Covered area of {0} on {1}'.format(cfg.region_model_id, oslo.to_string(time_x)))

fig, ax = plt.subplots(figsize=(15,5))
cm = plt.cm.get_cmap(color[1]) # color[0 to 75]
plot = ax.scatter(x, y, c=swe, vmin=swe_np.min(), vmax=swe_np.max(), marker='s', s=40, lw=0, cmap=cm)
plt.colorbar(plot)
plt.title('Snow Water Equivalent (mm) {0} on {1}'.format(cfg.region_model_id, oslo.to_string(time_x)))

fig, ax = plt.subplots(figsize=(15,5))
cm = plt.cm.get_cmap(color[72])# color[0 to 75]
plot = ax.scatter(x, y, c=sout, vmin=sout_np.min(), vmax=sout_np.max(), marker='s', s=40, lw=0, cmap=cm)
plt.colorbar(plot)
plt.title('Snow outflow {0} on {1}'.format(cfg.region_model_id, oslo.to_string(time_x)))
plt.show()


#   Histogram of SCA and SWE

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize = (15,6))
ax1.hist(sca, bins=20, range=(0,1), color='y', alpha=0.3)
ax1.set_xlabel("SCA of grid cell", fontsize=14)
ax1.set_ylabel("frequency", fontsize=14)

ax2.hist(swe, bins=20,  color='r', alpha=0.3)
ax2.set_xlabel("Snow Water Equivalent (mm) of grid cell", fontsize=14)
ax2.set_ylabel("frequency", fontsize=14)

plt.show()


#   Put SCA, SWE and outflow of all cells in  Pandas DataFrames and save them into CSV files

SCA_pd = geo_pd.copy()
SWE_pd = geo_pd.copy()
outflow_pd = geo_pd.copy()
dic_swe_ptgsk = {}
dic_sca_ptgsk = {}

ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
for day in range(0,cfg.number_of_steps):
    sca = simulator.region_model.gamma_snow_response.sca([],day)
    SCA_pd[ts_timestamps[day]] = sca
    dic_sca_ptgsk.update({day:sca})
SCA_pd.to_csv('SCA_pd.csv')

ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
for day in range(0,cfg.number_of_steps):
    swe = simulator.region_model.gamma_snow_response.swe([],day)
    SWE_pd[ts_timestamps[day]] = swe
    dic_swe_ptgsk.update({day:swe})
SWE_pd.to_csv('SWE_pd.csv')

outflow_pd = pd.DataFrame()
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
for day in range(0,cfg.number_of_steps):
    outflow = simulator.region_model.gamma_snow_response.outflow([],day)
    outflow_pd[ts_timestamps[day]] = outflow
outflow_pd.to_csv('outflow_pd.csv')

dic_swe_ptgsk_pd = pd.DataFrame(dic_swe_ptgsk)
dic_sca_ptgsk_pd = pd.DataFrame(dic_sca_ptgsk)

dic_swe_ptgsk_pd.to_csv('dic_swe_ptgsk_pd.csv')
dic_sca_ptgsk_pd.to_csv('dic_sca_ptgsk_pd.csv')

# SCA_pd = geo_pd.copy()
# SWE_pd = geo_pd.copy()
# outflow_pd = geo_pd.copy()
# dic_swe_pthsk = {}
# dic_sca_pthsk = {}

# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
#     sca = simulator.region_model.hbv_snow_state.sca([],day)
#     SCA_pd[ts_timestamps[day]] = sca
#     dic_sca_pthsk.update({day:sca})
# SCA_pd.to_csv('SCA_pd.csv')

# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
#     swe = simulator.region_model.hbv_snow_state.swe([],day)
#     SWE_pd[ts_timestamps[day]] = swe
#     dic_swe_pthsk.update({day:swe})
# SWE_pd.to_csv('SWE_pd.csv')

# outflow_pd = pd.DataFrame()
# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
#     outflow = simulator.region_model.hbv_snow_response.outflow([],day)
#     outflow_pd[ts_timestamps[day]] = outflow
# outflow_pd.to_csv('outflow_pd.csv')

# dic_swe_pthsk_pd = pd.DataFrame(dic_swe_pthsk)
# dic_sca_pthsk_pd = pd.DataFrame(dic_sca_pthsk)

# dic_swe_pthsk_pd.to_csv('dic_swe_pthsk_pd.csv')
# dic_sca_pthsk_pd.to_csv('dic_sca_pthsk_pd.csv')

# -----------------------------

# SCA_pd = geo_pd.copy()
# SWE_pd = geo_pd.copy()
# outflow_pd = geo_pd.copy()
# dic_swe_ptssk = {}
# dic_sca_ptssk = {}

# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
#     sca = simulator.region_model.skaugen_snow_state.sca([],day)
#     SCA_pd[ts_timestamps[day]] = sca
#     dic_sca_ptssk.update({day:sca})    
# SCA_pd.to_csv('SCA_pd.csv')

# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
#     swe = simulator.region_model.skaugen_snow_state.swe([],day)
#     SWE_pd[ts_timestamps[day]] = swe
#     dic_swe_ptssk.update({day:swe})
# SWE_pd.to_csv('SWE_pd.csv')

# outflow_pd = pd.DataFrame()
# ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in region_model.time_axis]
# for day in range(0,cfg.number_of_steps):
# #     outflow = simulator.region_model.gamma_snow_response.outflow([],day)
#     outflow = simulator.region_model.skaugen_snow_response.outflow([],day)
#     outflow_pd[ts_timestamps[day]] = outflow
# outflow_pd.to_csv('outflow_pd.csv')

# dic_swe_ptssk_pd = pd.DataFrame(dic_swe_ptssk)
# dic_sca_ptssk_pd = pd.DataFrame(dic_sca_ptssk)

# dic_swe_ptssk_pd.to_csv('dic_swe_ptssk_pd.csv')
# dic_sca_ptssk_pd.to_csv('dic_sca_ptssk_pd.csv')


#  Make SWE graph for a catchment or a cell in a period

while True:
    question = input("make a SWE graph, for a catchment or a cell?")
    if question == 'catchment' or question == 'cell' or question == 'stop':
        break
if question == 'catchment':

    print (region_model.catchment_ids)
    cid = input("Please enter catchment ID")
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))

    fig, ax = plt.subplots(figsize=(10,8))
    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
    swe_catch = simulator.region_model.gamma_snow_response.swe([int(cid)]).v.to_numpy()
    ax.plot(ts_timestamps,swe_catch[start_day:n_day+start_day], label = "{}".format(cid))
      
    fig.autofmt_xdate()
    ax.legend(title="Catch. ID")
    ax.set_ylabel("SWE (mm)")
    plt.show()
    print(swe_catch[start_day:n_day+start_day])
    
elif question == 'cell':
    
    print (f"Total number of cells are {simulator.region_model.size()}, enter from 0 to {simulator.region_model.size()-1}")
    cell_num = int(input("Enter the cell id"))
    start_day = int(input("start day (0 to {}) ?".format((cfg.number_of_steps-1))))
    left_days = cfg.number_of_steps - start_day
    n_day = int(input("how many days (0 to {}) ?".format(left_days)))    

    fig, ax = plt.subplots(figsize=(10,8))
    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(start_day),api.Calendar.DAY,n_day)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
    swe_cell = simulator.region_model.cells[cell_num].rc.snow_swe.v.to_numpy()
    ax.plot(ts_timestamps,swe_cell[start_day:n_day+start_day], label = f"Cell {cell_num}")
    
    fig.autofmt_xdate()       
    ax.legend(title="Cell ID")
    ax.set_ylabel("SWE (mm)")
    plt.show()
    print(swe_cell[start_day:n_day+start_day])

else:
    pass


#  Genarte all SCA images in whole period and save them in current directory

q2 = input("Do you want to genarte all SCA images in whole period ? ")
if q2 == 'yes':
    for idx in range(365):
        tim_x = 1377986400+2*3600 + idx*86400
        sca = simulator.region_model.gamma_snow_response.sca([],idx)
        fig, ax = plt.subplots(figsize=(25,11))
        cm = plt.cm.get_cmap(color[2]) # color[0 to 75]
        plot = ax.scatter(x, y, c=sca, vmin=0, vmax=1, marker='s', s=100, lw=0, cmap=cm)
        plt.colorbar(plot)
        plt.title('Snow Covered area of {0} on {1}'.format(cfg.region_model_id, dt.datetime.utcfromtimestamp(tim_x).date()),fontsize = 22)

        plt.savefig(f"SCA{idx}.png")
else:
    pass


#   Genarte all SWE images in whole period and save them in current directory

max_swe = 0

q2 = input("Do you want to genarte all SWE images in whole period ? ")
if q2 == 'yes':

    for idx in range(365):
        tim_x = 1377986400+2*3600 + idx*86400
        swe = simulator.region_model.gamma_snow_response.swe([],idx)
        
        swe_np = np.array(swe)
        if swe_np.max() > max_swe:
            max_swe = swe_np.max()
            
    for idx in range(365):
        tim_x = 1377986400+2*3600 + idx*86400
        swe = simulator.region_model.gamma_snow_response.swe([],idx)        
        
        
        fig, ax = plt.subplots(figsize=(25,11))
        cm = plt.cm.get_cmap(color[1]) # color[0 to 75]
        plot = ax.scatter(x, y, c=swe, vmin=0, vmax=max_swe, marker='s', s=100, lw=0, cmap=cm)
        plt.colorbar(plot)
        plt.title('Snow Water Equivalent (mm) {0} on {1}'.format(cfg.region_model_id, dt.datetime.utcfromtimestamp(tim_x).date()),fontsize = 22)

        plt.savefig(f"SWE{idx}.png")
        
else:
    pass


#  Discharge graphs of all targets and sum of all targets for the whole period for comparing the simulated ones and Ob. Ones with NSE

discharge_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\discharge.nc'

discharge_data = Dataset(discharge_file)

dis_pd = pd.DataFrame(np.array(discharge_data['discharge'][:]))
startdatetime = (int(str(cfg.start_datetime - datetime.datetime(2012,9,1)).split()[0])-1)

dis_target1 = dis_pd[0][:]
dis_target2 = dis_pd[1][:]
dis_target3 = dis_pd[2][:]
dis_targets = dis_pd[0][:] + dis_pd[1][:] + dis_pd[2][:]

dis_target1_np = np.array(dis_target1[startdatetime:cfg.number_of_steps+startdatetime])
dis_target2_np = np.array(dis_target2[startdatetime:cfg.number_of_steps+startdatetime])
dis_target3_np = np.array(dis_target3[startdatetime:cfg.number_of_steps+startdatetime])
dis_targets_np = np.array(dis_targets[startdatetime:cfg.number_of_steps+startdatetime])

target1 = [1308, 1394, 1867, 2198, 2402, 2545]
target2 = [1228, 1443, 1726, 2041, 2129, 2195, 2277, 2465, 2718, 3002, 3630, 1000010, 1000011]
target3 = [1996, 2446, 2640, 3536]

cid_z_map2 = {}
for key in cid_z_map.keys():
    if key in target1:
        cid_z_map2.update({key:1})
    elif key in target2:
        cid_z_map2.update({key:2})
    elif key in target3:
        cid_z_map2.update({key:3})
    else:
        cid_z_map2.update({key:0})
        
catch_ids2 = np.array([cid_z_map2[cell.geo.catchment_id()] for cell in cells])

# ---------------------- Target1 --------------------------------------- 

dis_sim1 = region_model.statistics.discharge(target1).v.to_numpy() # black

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,731)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

ax.plot(ts_timestamps, dis_sim1, lw=1.5, ls ='-', color = 'black',  label = f'Sim from simulator {target1}')
ax.plot(ts_timestamps, dis_target1_np, lw=1.5, ls ='-', color = 'red',  label = 'Obs. Directly from discharge.nc')

NSE1 = 1-(((dis_sim1-dis_target1_np)**2).sum()/((dis_target1_np-dis_target1_np.mean())**2).sum())

fig.autofmt_xdate()
ax.legend(title="Discharge", fontsize = 16, loc = 2)
ax.set_ylabel("discharge [m3 s-1]")
ax.set_title(f'Target 1, NSE1 = {round(NSE1,2)}', fontsize = 22)
plt.show()

# ---------------------- Target2 ---------------------------------------  

dis_sim2 = region_model.statistics.discharge(target2).v.to_numpy() # black

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,731)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

ax.plot(ts_timestamps, dis_sim2, lw=1.5, ls ='-', color = 'black',  label = f'Sim from simulator {target2}')
ax.plot(ts_timestamps, dis_target2_np, lw=1.5, ls ='-', color = 'red',  label = 'Obs. Directly from discharge.nc')

NSE2 = 1-(((dis_sim2-dis_target2_np)**2).sum()/((dis_target2_np-dis_target2_np.mean())**2).sum())

fig.autofmt_xdate()
ax.legend(title="Discharge", fontsize = 16, loc = 2)
ax.set_ylabel("discharge [m3 s-1]")
ax.set_title(f'Target 2, NSE2 = {round(NSE2,2)}', fontsize = 22)

plt.show()

# ---------------------- Target3 --------------------------------------- 

dis_sim3 = region_model.statistics.discharge(target3).v.to_numpy() # black

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,731)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

ax.plot(ts_timestamps, dis_sim3, lw=1.5, ls ='-', color = 'black',  label = f'Sim from simulator {target3}')
ax.plot(ts_timestamps, dis_target3_np, lw=1.5, ls ='-', color = 'red',  label = 'Obs. Directly from discharge.nc')

NSE3 = 1-(((dis_sim3-dis_target3_np)**2).sum()/((dis_target3_np-dis_target3_np.mean())**2).sum())

fig.autofmt_xdate()
ax.legend(title="Discharge", fontsize = 16, loc = 2)
ax.set_ylabel("discharge [m3 s-1]")
ax.set_title(f'Target 3, NSE3 = {round(NSE3,2)}', fontsize = 22)
plt.show()

# ---------------------- Targets ---------------------------------------  

dis_sims = dis_sim1 + dis_sim2 + dis_sim3

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,cfg.number_of_steps)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

ax.plot(ts_timestamps, dis_sims, lw=1, ls ='-', color = 'black',  label = 'Sim. discharge')
ax.plot(ts_timestamps, dis_targets_np, lw=1, ls ='-', color = 'red',  label = 'Obs. discharge')

NSEs = 1-(((dis_sims-dis_targets_np)**2).sum()/((dis_targets_np-dis_targets_np.mean())**2).sum())

fig.autofmt_xdate()
ax.legend(title="Discharge", fontsize = 16, loc = 2)
ax.set_ylabel("discharge [m3 s-1]")
ax.set_title(f'Targets, NSE = {round(NSEs,2)}', fontsize = 22)
plt.show()

# ---------------------- Catchments & Targets --------------------------- 

fig, ax = plt.subplots(figsize=(30,10))
cm = plt.cm.get_cmap(color[46])# color[0 to 75]
plot = ax.scatter(x, y, c=catch_ids2, marker='s',vmin = 0, vmax = 3, s=30, lw=5, cmap=cm)
# plot = ax.scatter(x, y, c=catch_ids2, marker='s',vmin = 0, vmax = 3, s=30, lw=5, cmap=cm)
plot = ax.scatter(x[140], y[140], marker='s',vmin = 0, vmax = 3, s=50, lw=0, cmap=cm, label ="Target 1", color = 'slateblue')
plot = ax.scatter(x[140], y[140], marker='s',vmin = 0, vmax = 3, s=50, lw=0, cmap=cm, label ="Target 2", color = 'deeppink')
plot = ax.scatter(x[140], y[140], marker='s',vmin = 0, vmax = 3, s=50, lw=0, cmap=cm, label ="Target 3", color = 'maroon')

# plot = ax.scatter(x, y, c=z, marker='o', s=10, lw=5, cmap=cm)
# plt.colorbar(plot).set_label('sub-catchments assocciate to targets')
plt.legend( fontsize = 16, loc = 1)
plt.show()

timelist = []
for i in range(len(ts_timestamps)):
    timelist.append((str(ts_timestamps[i])[0:10],dis_sims[i]))
timelist_pd = pd.DataFrame(timelist)
timelist_pd.to_csv('dis_sims_G.csv')


#  Somthing more to know
#  Getting access to defualt values of variables (not used in simulation)

parameterg = api.GammaSnowParameter()
parameterk = api.KirchnerParameter()
print('slow_albedo_decay_rate = ', parameterg.slow_albedo_decay_rate)
print('Kirchner C1 = ', parameterk.c1)


#  Getting access to the values are used for simulation which are in model.yaml (used in simulation)

param = simulator.region_model.get_region_parameter()
print('slow_albedo_decay_rate = ',param.gs.slow_albedo_decay_rate)
print('Kirchner C1 = ', param.kirchner.c1)


#  Getting access to atributes of simulator

for attr in dir(simulator.region_model):
    if attr[0] is not '_': #ignore privates
        print(attr)


#  Precipitation graph

precipitation_r = simulator.region_model.statistics.precipitation([])
precipitation_r_np = precipitation_r.values.to_numpy()

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,cfg.number_of_steps)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
ax.plot(ts_timestamps, precipitation_r_np, lw=1, ls ='-', color = 'red',  label = 'precipitation')
ax.legend(fontsize = 16, loc = 2)

precipitation_r_np_pd = pd.DataFrame(precipitation_r_np)
precipitation_r_np_pd.to_csv('precipitation_r_np_pd_G.csv')


#  Average SWE

SWE_average = simulator.region_model.gamma_snow_response.swe([])
SWE_average_np = ss1.values.to_numpy()

fig, ax = plt.subplots(figsize=(30,10))
ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,cfg.number_of_steps)   
ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]
ax.plot(ts_timestamps, SWE_average_np, lw=1, ls ='-', color = 'blue',  label = 'SWE_average')
ax.legend(fontsize = 16, loc = 2)

SWE_average_np_pd = pd.DataFrame(SWE_average_np)
SWE_average_np_pd.to_csv('SWE_average_np_pd_G.csv')


#  Calculation of the spent time

t2 = time.time()
t3 = t2-t1
hour1 = int(t3//3600)
minute1 = int((t3 % 3600)//60)
second1 = int(t3 - hour1*3600 - minute1*60)
print("",hour1,"Hours\n",minute1,"Minutes\n",second1,"Seconds")


#  Notify the end of simulation with an alarm

print('_It is done_'*7)
import winsound
for i in range(2500,3500,250):
    winsound.Beep(i, 850)
