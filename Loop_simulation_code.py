'''
This code does a loop simulation and read data from a CSV file
column by column and generate discharge validation graph
'''

from netCDF4 import Dataset
import os
import time
from os import path
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
for column in range(1,20):
    my_data = pd.read_csv(r"D:\Dropbox\Thesis\SHyFT\Gamma_parameters.csv")

    with open(r"D:\Dropbox\Thesis\SHyFT\neanidelva_model.yaml", 'w') as parameters2:
        print(f"model_t: !!python/name:shyft.api.pt_gs_k.PTGSKModel  # model to construct", file=parameters2)
        print(f"model_parameters:", file=parameters2)
        print(f"  ae:  # actual_evapotranspiration", file=parameters2)
        print(f"    ae_scale_factor: {my_data.iloc[3][c]}", file=parameters2)
        print(f"  gs:  # gamma_snow", file=parameters2)
        print(f"    calculate_iso_pot_energy: false", file=parameters2)
        print(f"    fast_albedo_decay_rate: {my_data.iloc[8][c]}", file=parameters2)
        print(f"    glacier_albedo: {my_data.iloc[15][c]}", file=parameters2)
        print(f"    initial_bare_ground_fraction: {my_data.iloc[21][c]}", file=parameters2)
        print(f"    max_albedo: {my_data.iloc[11][c]}", file=parameters2)
        print(f"    max_water: {my_data.iloc[6][c]}", file=parameters2)
        print(f"    min_albedo: {my_data.iloc[12][c]}", file=parameters2)
        print(f"    n_winter_days: {int(my_data.iloc[28][c])}", file=parameters2)
        print(f"    slow_albedo_decay_rate: {my_data.iloc[9][c]}", file=parameters2)
        print(f"    snow_cv: {my_data.iloc[14][c]}", file=parameters2)
        print(f"    snow_cv_altitude_factor: {my_data.iloc[18][c]}", file=parameters2)
        print(f"    snow_cv_forest_factor: {my_data.iloc[17][c]}", file=parameters2)
        print(f"    tx: {my_data.iloc[4][c]}", file=parameters2)
        print(f"    snowfall_reset_depth: {my_data.iloc[13][c]}", file=parameters2)
        print(f"    surface_magnitude: {my_data.iloc[10][c]}", file=parameters2)
        print(f"    wind_const: {my_data.iloc[7][c]}", file=parameters2)
        print(f"    wind_scale: {my_data.iloc[5][c]}", file=parameters2)
        print(f"    winter_end_day_of_year: {int(my_data.iloc[22][c])}", file=parameters2)
        print(f"  kirchner:", file=parameters2)
        print(f"    c1: {my_data.iloc[0][c]}", file=parameters2)
        print(f"    c2: {my_data.iloc[1][c]}", file=parameters2)
        print(f"    c3: {my_data.iloc[2][c]}", file=parameters2)
        print(f"  p_corr:  # precipitation_correction", file=parameters2)
        print(f"    scale_factor: {my_data.iloc[16][c]}", file=parameters2)
        print(f"  pt:  # priestley_taylor", file=parameters2)
        print(f"    albedo: {my_data.iloc[19][c]}", file=parameters2)
        print(f"    alpha: {my_data.iloc[20][c]}", file=parameters2)
        print(f"  routing:", file=parameters2)
        print(f"    alpha: {my_data.iloc[26][c]}", file=parameters2)
        print(f"    beta: {my_data.iloc[27][c]}", file=parameters2)
        print(f"    velocity: {my_data.iloc[25][c]}", file=parameters2)
        print(f"  gm:", file=parameters2)
        print(f"    direct_response: {my_data.iloc[29][c]}", file=parameters2)

# for column in range(1,20):
#     my_data = pd.read_csv(r"D:\Dropbox\Thesis\SHyFT\HBV_parameters.csv")

#     with open(r"D:\Dropbox\Thesis\SHyFT\neanidelva_model.yaml", 'w') as parameters2:
#         print(f"model_t: !!python/name:shyft.api.pt_hs_k.PTHSKModel  # priestley_taylor HBV_Snow kirchner", file=parameters2)
#         print(f"model_parameters:", file=parameters2)
#         print(f"  ae:  # actual_evapotranspiration", file=parameters2)
#         print(f"    ae_scale_factor: {my_data.iloc[3][column]}", file=parameters2)
#         print(f"  hs:  # HBV_Snow", file=parameters2)
#         print(f"    cfr: {my_data.iloc[8][column]}", file=parameters2)
#         print(f"    cx: {my_data.iloc[6][column]}", file=parameters2)
#         print(f"    lw: {my_data.iloc[4][column]}", file=parameters2)
#         print(f"    ts: {my_data.iloc[7][column]}", file=parameters2)
#         print(f"    tx: {my_data.iloc[5][column]}  ", file=parameters2)
#         print(f"  kirchner:", file=parameters2)
#         print(f"    c1: {my_data.iloc[0][column]}", file=parameters2)
#         print(f"    c2: {my_data.iloc[1][column]}", file=parameters2)
#         print(f"    c3: {my_data.iloc[2][column]}", file=parameters2)
#         print(f"  p_corr:  # precipitation_correction", file=parameters2)
#         print(f"    scale_factor: {my_data.iloc[10][column]}", file=parameters2)
#         print(f"  pt:  # priestley_taylor", file=parameters2)
#         print(f"    albedo: {my_data.iloc[11][column]}", file=parameters2)
#         print(f"    alpha: {my_data.iloc[12][column]}", file=parameters2)
#         print(f"  routing:", file=parameters2)
#         print(f"    alpha: {my_data.iloc[14][column]}", file=parameters2)
#         print(f"    beta: {my_data.iloc[15][column]}", file=parameters2)
#         print(f"    velocity: {my_data.iloc[13][column]}", file=parameters2)
             
# for column in range(1,20):
#     my_data = pd.read_csv(r"D:\Dropbox\Thesis\SHyFT\Skaugen_parameters.csv")

#     with open(r"D:\Dropbox\Thesis\SHyFT\neanidelva_model.yaml", 'w') as parameters2:
#         print(f"model_t: !!python/name:shyft.api.pt_ss_k.PTSSKModel  # priestley_taylor Skaugen_Snow kirchner", file=parameters2)
#         print(f"model_parameters:", file=parameters2)
#         print(f"  ae:  # actual_evapotranspiration", file=parameters2)
#         print(f"    ae_scale_factor: {my_data.iloc[3][column]}", file=parameters2)
#         print(f"  ss:  # Skaugen_Snow", file=parameters2)
#         print(f"    alpha_0: {my_data.iloc[4][column]}", file=parameters2)
#         print(f"    cfr: {my_data.iloc[11][column]}", file=parameters2)
#         print(f"    cx: {my_data.iloc[9][column]}", file=parameters2)
#         print(f"    d_range: {my_data.iloc[5][column]}", file=parameters2)
#         print(f"    max_water_fraction: {my_data.iloc[7][column]}", file=parameters2)
#         print(f"    ts: {my_data.iloc[10][column]}", file=parameters2)
#         print(f"    tx: {my_data.iloc[8][column]}", file=parameters2)
#         print(f"    unit_size: {my_data.iloc[6][column]}", file=parameters2)
#         print(f"  kirchner:", file=parameters2)
#         print(f"    c1: {my_data.iloc[0][column]}", file=parameters2)
#         print(f"    c2: {my_data.iloc[1][column]}", file=parameters2)
#         print(f"    c3: {my_data.iloc[2][column]}", file=parameters2)
#         print(f"  p_corr:  # precipitation_correction", file=parameters2)
#         print(f"    scale_factor: {my_data.iloc[12][column]}", file=parameters2)
#         print(f"  pt:  # priestley_taylor", file=parameters2)
#         print(f"    albedo: {my_data.iloc[13][column]}", file=parameters2)
#         print(f"    alpha: {my_data.iloc[14][column]}", file=parameters2)
#         print(f"  routing:", file=parameters2)
#         print(f"    alpha: {my_data.iloc[17][column]}", file=parameters2)
#         print(f"    beta: {my_data.iloc[18][column]}", file=parameters2)
#         print(f"    velocity: {my_data.iloc[16][column]}", file=parameters2)

    time.sleep(5)
        
    shyft_data_path = path.abspath(r"C:\shyft_workspace\shyft-data")
    if path.exists(shyft_data_path) and 'SHYFT_DATA' not in os.environ:
        os.environ['SHYFT_DATA']=shyft_data_path
        
    import shyft
    from shyft import api
    from shyft.repository.default_state_repository import DefaultStateRepository
    from shyft.orchestration.configuration.yaml_configs import YAMLSimConfig
    from shyft.orchestration.simulators.config_simulator import ConfigSimulator
    
    config_file_path = r'D:\Dropbox\Thesis\SHyFT\neanidelva_simulation.yaml'
    cfg = YAMLSimConfig(config_file_path, "neanidelva")
    
    simulator = ConfigSimulator(cfg)
    region_model = simulator.region_model
    
    simulator.region_model.set_snow_sca_swe_collection(-1,True)
    simulator.region_model.set_state_collection(-1,True)
    simulator.run()    

    cells = region_model.get_cells()

    x = np.array([cell.geo.mid_point().x for cell in cells])
    y = np.array([cell.geo.mid_point().y for cell in cells])
    z = np.array([cell.geo.mid_point().z for cell in cells])
    area = np.array([cell.geo.area() for cell in cells])
    catch_ids = np.array([cell.geo.catchment_id() for cell in cells])      
    
    catchment_ids = region_model.catchment_ids
    
    cid_z_map = dict([(catchment_ids[i],i) for i in range(len(catchment_ids))])
    print(cid_z_map)

    catch_ids = np.array([cid_z_map[cell.geo.catchment_id()] for cell in cells])
    
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

    dis_sim1 = region_model.statistics.discharge(target1).v.to_numpy() 
    dis_sim2 = region_model.statistics.discharge(target2).v.to_numpy() 
    dis_sim3 = region_model.statistics.discharge(target3).v.to_numpy() 
    dis_sims = dis_sim1 + dis_sim2 + dis_sim3

    fig, ax = plt.subplots(figsize=(30,10))
    ta_statistics = api.TimeAxis(simulator.region_model.time_axis.time(0),api.Calendar.DAY,cfg.number_of_steps)   
    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in ta_statistics]

    ax.plot(ts_timestamps, dis_sims, lw=1, ls ='-', color = 'black',  label = 'Sim from simulator for all targets')
    ax.plot(ts_timestamps, dis_targets_np, lw=1, ls ='-', color = 'red',  label = 'Obs. Directly from discharge.nc')

    NSEs = 1-(((dis_sims-dis_targets_np)**2).sum()/((dis_targets_np-dis_targets_np.mean())**2).sum())

    fig.autofmt_xdate()
    ax.legend(title="Discharge", fontsize = 16, loc = 2)
    ax.set_ylabel("discharge [m3 s-1]")
    ax.set_title(f'Targets, NSE = {round(NSEs,2)}', fontsize = 22)

    file_name = str(column)
    plt.savefig(f"D:\\Dropbox\\Thesis\\{file_name}.png")

    plt.show()

print('_It is done_'*7)
import winsound
for i in range(2500,3500,250):
    winsound.Beep(i, 850)    
