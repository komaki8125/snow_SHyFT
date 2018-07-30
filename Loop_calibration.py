'''
The typical calibration in SHyFT is a one-time calibration and after it is done it stops and the calibrated
parameters are saved in a calibrated.yaml file and it is needed to rerun the calibration code to get another
calibrated parameters. Also for the validation part it needs to modify the parameters in the model.yaml manually
then run the simulation code. In this study we decided to have 200 calibrations for each method so I add more python
codes and make a loop calibration. The following code do calibration forever and save the calibrated parameters in
a CSV file and update it after a new loop. Also it generates the simulated and observed discharge graph for every
calibration time and save in a folder. So it is possible to have many calibrated parameters in a single CSV file and
their graphs after some days. This CSV parameters file will be used in validation part without needs to modify
the model.yaml manually. This calibration code can be used for all type of methods.
'''
# Importing the third-party python modules

import os
from os import path
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import random

all_results, good_results = [], []
counter = -1

while counter < 0:
    shyft_data_path = path.abspath(r"C:\shyft_workspace\shyft-data")
    if path.exists(shyft_data_path) and 'SHYFT_DATA' not in os.environ:
        os.environ['SHYFT_DATA']=shyft_data_path
    from shyft.repository.default_state_repository import DefaultStateRepository
    from shyft.orchestration.configuration.yaml_configs import YAMLCalibConfig, YAMLSimConfig
    from shyft.orchestration.simulators.config_simulator import ConfigCalibrator, ConfigSimulator  
    
    counter += 1
    t1 = time.time()

    config_file_path = os.path.abspath(r"D:\Dropbox\Thesis\SHyFT\Yaml_files\Skaugen\neanidelva_simulation.yaml")
    cfg = YAMLSimConfig(config_file_path, "neanidelva")
    simulator = ConfigSimulator(cfg)
    simulator.run()
    state = simulator.region_model.state
    region_model = simulator.region_model
    
    config_file_path = os.path.abspath(r"D:\Dropbox\Thesis\SHyFT\Yaml_files\Skaugen\neanidelva_simulation.yaml")
    cfg = YAMLCalibConfig(config_file_path, "neanidelva")
    
    calib = ConfigCalibrator(cfg)
    cfg.optimization_method['params']['tr_start'] = random.randrange(1,2000)/10000
        
    state_repos = DefaultStateRepository(calib.region_model)
    results = calib.calibrate(cfg.sim_config.time_axis, state_repos.get_state(0).state_vector,
                              cfg.optimization_method['name'], cfg.optimization_method['params'])
    t2 = time.time()
    now = str(dt.datetime.now())
    result_params = []
    for i in range(results.size()):
        result_params.append(results.get(i))

    result_params.append(1-calib.optimizer.calculate_goal_function(result_params))
    
    result_params.append(int((t2-t1)/60))
    result_params.append(now)
    result_params.append(str(cfg.optimization_method['name']))
    result_params.append(str(cfg.optimization_method['params']))
    result_params.append(str(region_model.time_axis)[-30:-20])
    result_params.append(str(str(region_model.time_axis).split(',')[-1][:-1]))
    result_params.append(str(cfg.overrides['model']['model_t'])[-15:-10])
    result_params.append(str(cfg.calibration_parameters))
    
    all_results.append(result_params)
             
    pd_results = pd.DataFrame(all_results)
    pd_good_results = pd.DataFrame(good_results)
             
    pd_results_2 = pd_results.transpose()
    pd_good_results_2 = pd_good_results.transpose()
    
    if str(cfg.overrides['model']['model_t'])[-15:-10] == 'PTGSK':
        param_list = ['kirchner.c1','kirchner.c2','kirchner.c3','ae.ae_scale_factor','gs.tx','gs.wind_scale','gs.max_water','gs.wind_const','gs.fast_albedo_decay_rate','gs.slow_albedo_decay_rate','gs.surface_magnitude','gs.max_albedo','gs.min_albedo','gs.snowfall_reset_depth','gs.snow_cv','gs.glacier_albedo','p_corr.scale_factor','gs.snow_cv_forest_factor','gs.snow_cv_altitude_factor','pt.albedo','pt.alpha','gs.initial_bare_ground_fraction','gs.winter_end_day_of_year','gs.calculate_iso_pot_energy','gm.dtf','routing.velocity','routing.alpha','routing.beta','gs.n_winter_days','gm.direct_response','NSE','Computation time(Minutes)','Date & Time','Method name','Params method','Start datetime','Number of days','Model name','Ranges']
    
    elif str(cfg.overrides['model']['model_t'])[-15:-10] == 'PTHSK':
        param_list = ['kirchner.c1','kirchner.c2','kirchner.c3','ae.ae_scale_factor','hs.lw','hs.tx','hs.cx','hs.ts','hs.cfr','gm.dtf','p_corr.scale_factor','pt.albedo','pt.alpha','routing.velocity','routing.alpha','routing.beta','gm.direct_response','NSE','Computation time(Minutes)','Date & Time','Method name','Params method','Start datetime','Number of days','Model name','Ranges']
  
    elif str(cfg.overrides['model']['model_t'])[-15:-10] == 'PTSSK':
        param_list = ['kirchner.c1','kirchner.c2','kirchner.c3','ae.ae_scale_factor','ss.alpha_0','ss.d_range','ss.unit_size','ss.max_water_fraction','ss.tx','ss.cx','ss.ts','ss.cfr','p_corr.scale_factor','pt.albedo','pt.alpha','gm.dtf','routing.velocity','routing.alpha','routing.beta','gm.direct_response','NSE','Computation time(Minutes)','Date & Time','Method name','Params method','Start datetime','Number of days','Model name','Ranges']

    pd_param = pd.DataFrame(param_list)
    pd_res = pd.concat([pd_param, pd_results_2], axis = 1)
     
    pd_res2 = pd.concat([pd_param, pd_good_results_2], axis = 1)            
                  
    pd_res.to_csv('D:\\Dropbox\\Thesis\\SHyFT\\Results\\results.csv')
    
    target_obs = calib.tv[0]
    disch_sim_all = np.linspace(0,0,target_obs.ts.time_axis.size())
    disch_obs_all = np.linspace(0,0,target_obs.ts.time_axis.size())

    for tar in range(calib.tv.size()):

        target_obs = calib.tv[tar]
        disch_sim = calib.region_model.statistics.discharge(target_obs.catchment_indexes)
        disch_obs = target_obs.ts.values
        disch_sim_np = np.array(disch_sim.values)  
        disch_obs_np = np.array(disch_obs)
        disch_sim_all += disch_sim_np
        disch_obs_all += disch_obs_np

    ts_timestamps = [dt.datetime.utcfromtimestamp(p.start) for p in target_obs.ts.time_axis]

    fig, ax = plt.subplots(1, figsize=(45,10))
    ax.plot(ts_timestamps, disch_sim_all, lw=1, ls = '-', label = "sim", color = 'navy')
    ax.plot(ts_timestamps, disch_obs_all, lw=1, ls='-', label = "obs", color = 'crimson')
    ax.set_title(f"observed and simulated discharge (sum of all catchments) {str(simulator.region_model.__class__)[-12:-7]}")
    ax.legend()
    ax.set_ylabel("discharge [m3 s-1]")

    plt.savefig(f"D:\\Dropbox\\Thesis\\SHyFT\\Results\\{counter}.png")
