from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os


# 1. Precipitation

precipitation_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\precipitation.nc'

precipitation_data = Dataset(precipitation_file)

series_pd = pd.DataFrame()
pre_pd = pd.DataFrame(np.array(precipitation_data['precipitation'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(precipitation_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, pre_pd]
precipitation = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(precipitation_file))
# get the file name without extension
file_name = precipitation_file.split('\\')[-1].split(".")[-2]

precipitation.to_csv(f'{file_name}.csv')


# 2. GeoCell

cells_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\cell_data.nc'
cell_data = Dataset(cells_file)
 
cells_pd = pd.DataFrame()

for key in cell_data.variables.keys():
    cells_pd[key] = np.array(cell_data[key][:])
    
# set the current directory to the file directory
os.chdir(os.path.dirname(cells_file))
# get the file name without extension
file_name = cells_file.split('\\')[-1].split(".")[-2]

cells_pd.to_csv(f'{file_name}.csv')


# 3. Disharge

discharge_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\discharge.nc'

discharge_data = Dataset(discharge_file)

series_pd = pd.DataFrame()
dis_pd = pd.DataFrame(np.array(discharge_data['discharge'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(discharge_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, dis_pd]
discharge = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(discharge_file))
# get the file name without extension
file_name = discharge_file.split('\\')[-1].split(".")[-2]

discharge.to_csv(f'{file_name}.csv')


# 4. Radiation

radiation_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\radiation.nc'

radiation_data = Dataset(radiation_file)

series_pd = pd.DataFrame()
radi_pd = pd.DataFrame(np.array(radiation_data['global_radiation'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(radiation_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, radi_pd]
radiation = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(radiation_file))
# get the file name without extension
file_name = radiation_file.split('\\')[-1].split(".")[-2]

radiation.to_csv(f'{file_name}.csv')


# 5. Relative_humidity

humidity_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\relative_humidity.nc'

humidity_data = Dataset(humidity_file)

series_pd = pd.DataFrame()
humi_pd = pd.DataFrame(np.array(humidity_data['relative_humidity'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(humidity_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, radi_pd]
humidity = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(humidity_file))
# get the file name without extension
file_name = humidity_file.split('\\')[-1].split(".")[-2]

humidity.to_csv(f'{file_name}.csv')


# 6. Temperature

temperature_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\temperature.nc'

temperature_data = Dataset(temperature_file)

series_pd = pd.DataFrame()
temp_pd = pd.DataFrame(np.array(temperature_data['temperature'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(temperature_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, temp_pd]
temperature = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(temperature_file))
# get the file name without extension
file_name = temperature_file.split('\\')[-1].split(".")[-2]

temperature.to_csv(f'{file_name}.csv')


# 7. wind_speed

wind_file = r'C:\shyft_workspace\shyft-data\netcdf\orchestration-testdata\wind_speed.nc'

wind_data = Dataset(wind_file)

series_pd = pd.DataFrame()
wind_pd = pd.DataFrame(np.array(wind_data['wind_speed'][:]))

for item in ['x', 'y', 'z', 'series_name']:
    series_pd[repr(item)] = np.array(wind_data[item][:])
series_2_pd = series_pd.transpose()    
    
frames = [series_2_pd, wind_pd]
wind = pd.concat(frames)

# set the current directory to the file directory
os.chdir(os.path.dirname(wind_file))
# get the file name without extension
file_name = wind_file.split('\\')[-1].split(".")[-2]

wind.to_csv(f'{file_name}.csv')
