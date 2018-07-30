from netCDF4 import Dataset
import os
from os import path
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
color = {0:'viridis', 1:'plasma', 2:'inferno', 3:'magma', 4:'Greys',
5:'Purples', 6:'Blues', 7:'Greens',  8:'Oranges', 9:'Reds', 10:'YlOrBr'}


while True:

    dic_sca_ptgsk_pd2=pd.read_csv('dic_sca_ptgsk_pd.csv')
    dic_sca_pthsk_pd2=pd.read_csv('dic_sca_pthsk_pd.csv')
    dic_sca_ptssk_pd2=pd.read_csv('dic_sca_ptssk_pd.csv')
    
    dic_swe_ptgsk_pd2=pd.read_csv('dic_swe_ptgsk_pd.csv')
    dic_swe_pthsk_pd2=pd.read_csv('dic_swe_pthsk_pd.csv')
    dic_swe_ptssk_pd2=pd.read_csv('dic_swe_ptssk_pd.csv')    
    
    geo_data = pd.read_csv('geo_pd.csv')

    for idx in range(364): # one year
        tim_x = 1377986400+2*3600 + idx*86400 # 2013/9/1

        fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize = (25,13))

        cm = plt.cm.get_cmap(color[3]) # color[0 to 75]
        ax1.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-6830000,        c=dic_sca_ptgsk_pd2[str(idx)][:], vmin=0, vmax = 1, marker='s', s=100, lw=0, cmap=cm)
        ax1.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-60000-6830000,  c=dic_sca_pthsk_pd2[str(idx)][:], vmin=0, vmax = 1, marker='s', s=100, lw=0, cmap=cm)
        ax1.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-120000-6830000, c=dic_sca_ptssk_pd2[str(idx)][:], vmin=0, vmax = 1, marker='s', s=100, lw=0, cmap=cm)

        ax1.annotate('Gamma Snow (0 - 0.96)', xy =(5500,211000), fontsize = 16, color = "black")
        ax1.annotate('HBV Snow (0 - 1)', xy =(5500,152000), fontsize = 16, color = "black")
        ax1.annotate('Skaugen Snow (0 - 1)', xy =(5500,92000), fontsize = 16, color = "black")

        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ax1.set_title('Snow Cover Area on {}'.format(dt.datetime.utcfromtimestamp(tim_x).date()),fontsize = 20)

        cm = plt.cm.get_cmap(color[1]) # color[0 to 75]
        ax2.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-6830000, c=dic_swe_ptgsk_pd2[str(idx)][:], vmin=0, vmax = 600, marker='s', s=100, lw=0, cmap=cm)
        ax2.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-60000-6830000, c=dic_swe_pthsk_pd2[str(idx)][:], vmin=0, vmax = 25, marker='s', s=100, lw=0, cmap=cm)
        ax2.scatter(geo_data['x'][:]-260000, geo_data['y'][:]-120000-6830000, c=dic_swe_ptssk_pd2[str(idx)][:], vmin=0, vmax = 25, marker='s', s=100, lw=0, cmap=cm)

        ax2.annotate('Gamma Snow', xy =(5500,211000), fontsize = 16, color = "black")
        ax2.annotate('HBV Snow', xy =(5500,152000), fontsize = 16, color = "black")
        ax2.annotate('Skaugen Snow', xy =(5500,92000), fontsize = 16, color = "black")

        ax2.set_xticks([])
        ax2.set_yticks([])
 
        ax2.set_title('Snow Water Equivalent (mm) on {}'.format(dt.datetime.utcfromtimestamp(tim_x).date()),fontsize = 20)
        
        plt.savefig(f"SCA_PTGSK_PTHSK_PTSSK{idx}.png")
