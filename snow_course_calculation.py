'''
This Python Scripts, first import some Python module, Pandas; to make DataFarme, Numpy; for calculation,
Matplotlib; make graphs, OS and deepcopy. Then read the X, Y coordinates and SWE of all snow course lines
of all years between 2013 to 2017 from some CSV files. Then read X, Y and Z coordinates of the catchment.
Afterwards, calculate the distances between all snow course line points and the center points of all cells.
So it finds each snow course line points in which cell and then categorized all snow course line points of
a cell in an individual group. It Calculates the SWE average, minimum, maximum and standard division of each group.
Then it reads the date of doing snow course and find the SWE of that cell and compares with the SWE average
from snow course points in that cell. Also read the elevations of neighbor cells and calculates the elevation
gradient of some cells which include at least one snow course point. Then make three illustrations,
first shows the catchment shape and the snow course line position and its length. Second, shows snow course
lines and shows the center of cells with dots, if it doesn’t include some snow course line points it is a red dot
and if it includes some, it is black and shows its boundary. All latter cells include cell No., the number of snow
course line points, average, maximum, minimum, standard division, the elevation gradient and the orientation slope with
an arrow, the SWE of that cells and its accuracy. The third illustration is the SWE histogram of the snow course line
and the SWE of passed cells. Finally, write all this values in a CSV file. The Python code is in following as well
as three illustrations.
'''

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy

for snowcourse1 in range(4,10):
    for snowcourse2 in range(2013,2018): # years 2013, 2014, 2015, 2016, 2017
        snow_course_pd = pd.DataFrame()
        all_swe_pd = pd.DataFrame()        
        snow_course_file = r"D:\Dropbox\Thesis\Nea snowradar transects\NE0" + str(snowcourse1) + "_" + str(snowcourse2) + ".csv"

        # reading the simulated SWE data in all cells which is given out by simulation
        all_swe_pd = pd.read_csv(r'D:\Dropbox\Thesis\Nea snowradar transects\SWE_pd_18_G.csv') 

        # get the file name without extension
        file_name = snow_course_file.split('\\')[-1].split(".")[-2]

        # set the current directory to the where read the 'snow course file'
        os.chdir(os.path.dirname(snow_course_file)) 

        # make DataFrame for snow course
        snow_course_pd = pd.read_csv(snow_course_file)

        # the date of doing snow course 
        if file_name == 'NE01_2013' or file_name == 'NE02_2013' or file_name == 'NE03_2013': date_get = '11-Apr-13'    
        elif file_name == 'NE04_2013' or file_name == 'NE05_2013' or file_name == 'NE06_2013' or file_name == 'NE07_2013' or file_name == 'NE08_2013' or file_name == 'NE09_2013': date_get = '10-Apr-13'    

        elif file_name == 'NE01_2014' or file_name == 'NE02_2014' or file_name == 'NE04_2014': date_get = '2-Apr-14'
        elif file_name == 'NE03_2014': date_get = '9-Apr-14'
        elif file_name == 'NE05_2014' or file_name == 'NE06_2014' or file_name == 'NE07_2014' or file_name == 'NE08_2014' or file_name == 'NE09_2014': date_get = '1-Apr-14'

        elif file_name == 'NE01_2015' or file_name == 'NE03_2015' or file_name == 'NE04_2015' or file_name == 'NE09_2015': date_get = '9-Apr-15'    
        elif file_name == 'NE02_2015' or file_name == 'NE05_2015' or file_name == 'NE06_2015' or file_name == 'NE07_2015' or file_name == 'NE08_2015': date_get = '10-Apr-15'    

        elif file_name == 'NE01_2016' or file_name == 'NE03_2016' or file_name == 'NE04_2016' or file_name == 'NE09_2016': date_get = '4-Apr-16'    
        elif file_name == 'NE02_2016' or file_name == 'NE06_2016' or file_name == 'NE07_2016': date_get = '11-Apr-16'    
        elif file_name == 'NE05_2016' or file_name == 'NE08_2016': date_get = '5-Apr-16'    

        elif file_name == 'NE01_2017' or file_name == 'NE02_2017' or file_name == 'NE03_2017' or file_name == 'NE04_2017': date_get = '7-Mar-17'    
        elif file_name == 'NE05_2017' or file_name == 'NE06_2017' or file_name == 'NE07_2017' or file_name == 'NE08_2017' or file_name == 'NE09_2017': date_get = '8-Mar-17'    

        # to read the simulated SWE form a CSV file
        if date_get == '11-Apr-13': cell_swe_no = 228
        elif date_get == '10-Apr-13': cell_swe_no = 227

        elif date_get == '1-Apr-14': cell_swe_no = 583
        elif date_get == '2-Apr-14': cell_swe_no = 584
        elif date_get == '9-Apr-14': cell_swe_no = 591

        elif date_get == '9-Apr-15': cell_swe_no = 956
        elif date_get == '10-Apr-15': cell_swe_no = 957      

        elif date_get == '4-Apr-16': cell_swe_no = 1317
        elif date_get == '5-Apr-16': cell_swe_no = 1318
        elif date_get == '11-Apr-16': cell_swe_no = 1324    

        elif date_get == '7-Mar-17': cell_swe_no = 1654
        elif date_get == '8-Mar-17': cell_swe_no = 1655
            
        cells_x_np22 = np.array(all_swe_pd.loc[1:]['2'])
        cells_xs = []
        for item in cells_x_np22:
            cells_xs.append(float(item))
        cells_x_np = np.array(cells_xs)

        cells_y_np22 = np.array(all_swe_pd.loc[1:]['3'])
        cells_ys = []
        for item in cells_y_np22:
            cells_ys.append(float(item))
        cells_y_np = np.array(cells_ys)

        path_x_np = np.array(snow_course_pd[:]['X'])
        path_y_np = np.array(snow_course_pd[:]['Y'])
        swe_np = np.array(snow_course_pd[:]['SWE'])
        swe = list(swe_np)

        cells, path=[], [] 

        for i in range(len(cells_x_np)):
            cells.append((cells_x_np[i],cells_y_np[i]))

        for i in range(len(path_x_np)):
            path.append((path_x_np[i],path_y_np[i]))            

        snow_course_length = 0
        for i in range(len(path_x_np)-1):
            dd = ((path_x_np[i]-path_x_np[i+1])**2 + (path_y_np[i]-path_y_np[i+1])**2)**0.5
            snow_course_length += dd
        snow_course_length = round(snow_course_length,1)
        print('Lenght of snow course:\t\t ', snow_course_length, "meters")
        print('Number of cells in the catchment:', cells_x_np.size)            

        list1, distance =[], []

        for i in range(len(path)):
            for j in range(len(cells)):
                list1.append((((path[i][0])-(cells[j][0]))**2+((path[i][1])-(cells[j][1]))**2)**0.5)
            distance.append(list1)
            list1 = []            
            
        distance_pd = pd.DataFrame(distance)
        distance2_pd = distance_pd.transpose()
        first = distance_pd[:][0]
        cell_close_no = []
        for i in range(first.size):
            cell_close_no.append(list(distance2_pd[i][:]).index(distance2_pd[i][:].min()))

        list_close_cell = list(set(cell_close_no))  
        list_close_cell.sort()

        all_cat, averageif, std_swe, min_swe, max_swe, cv_swe, no_swe,forprint = [],[],[],[],[],[],[], []

        for j in range(len(set(cell_close_no))):
            sum, counter = 0,0
            all_cat.append([])
            for i in range(len(cell_close_no)):
                if list_close_cell[j] == cell_close_no[i]:
                    sum += swe[i]
                    counter += 1
                    all_cat[j].append(swe[i])
            averageif.append(sum/counter)
            new_np = np.array(all_cat[j])
            std_swe.append(new_np.std())
            min_swe.append(new_np.min())
            max_swe.append(new_np.max())
            cv_swe.append(new_np.std()/(sum/counter))
            no_swe.append(counter)
            forprint.append([])
            forprint[j].append(list_close_cell[j])
            forprint[j].append(counter)
            average_1 = sum/counter
            forprint[j].append(round(average_1,2))
            forprint[j].append(round(new_np.std(),2))
            forprint[j].append(round(new_np.min(),2))
            forprint[j].append(round(new_np.max(),2))

        for i in range(len(cell_close_no)-len(list_close_cell)):
            list_close_cell.append(0)
            averageif.append(0)
            std_swe.append(0)
            min_swe.append(0)
            max_swe.append(0)
            cv_swe.append(0)    
            no_swe.append(0)

        snow_course_pd['inside_cell'] = cell_close_no
        snow_course_pd['Cell No.'] = list_close_cell
        snow_course_pd['AverageIf'] = averageif
        snow_course_pd['Minimum'] = min_swe
        snow_course_pd['Maximum'] = max_swe
        snow_course_pd['Standard Deviation'] = std_swe
        snow_course_pd['CV'] = cv_swe
        snow_course_pd['No.'] = no_swe
        snow_course_pd.to_csv(f'{file_name}_cell_close_no.csv')            

        fig, ax1 = plt.subplots(figsize=(25,9))

        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(12)   

        close = []
        for i in range(len(cells_x_np)):
            close.append(distance_pd[:][i].min())    

        cm = plt.cm.get_cmap('tab20c')
        ax1.scatter(cells_x_np, cells_y_np, c=close, marker='o', s=220, lw=0, cmap=cm, alpha = 0.9)

        ax1.plot(path_x_np, path_y_np, lw = 1.4, color = 'black', label=f'Snow course line ({snow_course_length} Meters)')

        file_name = snow_course_file.split('\\')[-1].split(".")[-2]                

        plt.title(f"Snow course on the catchment layout ({date_get}): {file_name}", fontsize = 14)  
        plt.xlabel('X coordinade')
        plt.ylabel('Y coordinade')
        ax1.legend(loc=1, fontsize = 14)

        plt.savefig(f"{file_name}_1.png")
        plt.show()            
            
        draw_all_cells = 'no' # 'yes' or 'no'
        range_cell = 650
        font_size = 14

        cells_x_2, cells_y_2 = [],[]

        for i in range (len(cells_x_np)):
            if cells_x_np[i] > path_x_np.min()-range_cell and cells_x_np[i] < path_x_np.max()+range_cell:
                if cells_y_np[i] > path_y_np.min()-range_cell and cells_y_np[i] < path_y_np.max()+range_cell:
                    cells_x_2.append(cells_x_np[i])
                    cells_y_2.append(cells_y_np[i])

        cells_x_np2 = np.array(cells_x_2)
        cells_y_np2 = np.array(cells_y_2)

        list1 = []
        for i in range(len(cells_x_np2)):
            for j in range(len(cells_x_np)):
                       if cells_x_np2[i] == cells_x_np[j]:
                           if cells_y_np2[i] == cells_y_np[j]:
                                list1.append([cells_x_np2[i],cells_y_np2[i],j])

        status = 0
        for i in range(len(list1)):
            for j in range(len(forprint)):
                if list1[i][2] == forprint[j][0]:
                    list1[i].append(forprint[j][1])
                    list1[i].append(forprint[j][2])
                    list1[i].append(forprint[j][3])
                    list1[i].append(forprint[j][4])
                    list1[i].append(forprint[j][5])
                    status +=1
            if status == 0:
                list1[i].append(0)
                list1[i].append(0)
                list1[i].append(0)
                list1[i].append(0)
                list1[i].append(0)
            status = 0               

        # make a deep copy of list1
        list2 = deepcopy(list1)
        cell_list = []

        for i in range(cells_x_np.size):
            cell_list_temp = [float(all_swe_pd.loc[i+1][1]),float(all_swe_pd.loc[i+1][2]),float(all_swe_pd.loc[i+1][3])]
            cell_list.append(cell_list_temp)

        for i in range(len(list2)):
            for j in range(len(cell_list)):
                if int(list2[i][0]) == int(cell_list[j][0]):
                    if int(list2[i][1]) == int(cell_list[j][1]):
                        list2[i].append(cell_list[j][2])

        list_x_new, list_y_new, list_z_new, distan_new, gradia_new = [], [], [], [], []

        for i in range(len(list2)):
            for j in range(len(cell_list)):
                if int(cell_list[j][0]) < int(list2[i][0]) + 1500 and int(cell_list[j][0]) > int(list2[i][0])-1500:
                    if int(cell_list[j][1]) < int(list2[i][1]) + 1500 and int(cell_list[j][1]) > int(list2[i][1])-1500:
                        list_x_new.append(cell_list[j][0])
                        list_y_new.append(cell_list[j][1])
                        list_z_new.append(cell_list[j][2])

            for l in range(len(list_x_new)):
                dis = ((list2[i][0]-list_x_new[l])**2 + (list2[i][1]-list_y_new[l])**2)**0.5
                if dis == 0:
                    gradian = 0
                    gradia_new.append(gradian)
                    distan_new.append(dis)
                    continue
                gradian = (list2[i][8] - list_z_new[l]) / dis
                distan_new.append(dis)
                gradia_new.append(gradian)
            gradia_new_np = np.array(np.abs(gradia_new))

            gr = 0
            for ii in range(len(gradia_new_np)):
                if gradia_new[ii] > gr:
                    gr = gradia_new_np[ii]
                    x_compare = list_x_new[ii]
                    y_compare = list_y_new[ii]

            if list2[i][0] == x_compare:
                if list2[i][1] < y_compare:
                    o ="North"
                elif list2[i][1] > y_compare:
                    o ="South"
            elif list2[i][0] < x_compare:
                if list2[i][1] < y_compare:
                    o ='North-East'
                elif list2[i][1] == y_compare:
                    o ='Easth'
                else:
                    o ='South-East'
            elif list2[i][0] > x_compare:
                if list2[i][1] < y_compare:
                    o ='North-West'
                elif list2[i][1] == y_compare:
                    o ='West'
                else:
                    o ='South-West'
					
        #    list2[i].append(int(gradia_new_np.mean()*100)) # take the average slope of the cells       
            list2[i].append(int(gradia_new_np.max()*100)) # take the maximum slope of the cells
            list2[i].append(o)
            list_x_new, list_y_new, list_z_new, distan_new, gradia_new = [], [], [], [], []

        grad = []
        orientation = []
        for i in range(len(list2)):
            if list2[i][3]!=0:
                grad.append(list2[i][9])
                orientation.append(list2[i][10])

        for i in range(len(cell_close_no)-len(grad)):
            grad.append(0)
            orientation.append(0)

        snow_course_pd['Elevation gradient'] = grad
        snow_course_pd['Orientation'] = orientation

        snow_course_pd.to_csv(f'{file_name}_cell_close_no.csv')

        for i in range(len(list2)):
            if list2[i][3] != 0:
                list2[i].append(all_swe_pd.loc[list2[i][2]+1][cell_swe_no])
            else:
                list2[i].append(0)            

        fig, ax = plt.subplots(figsize=(int((max(cells_x_2) - min(cells_x_2)+1000)/270),
                                        int((max(cells_y_2) - min(cells_y_2) + 1000)/270)))

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

        label_stat1, label_stat2 = True, True

        for d in range(len(list1)):
            if list1[d][3] == 0:
                if label_stat1:
                    plot = ax.scatter(list1[d][0], list1[d][1], marker='o', s=60, lw=0, color = 'red',
                                      label = 'snow course doesn`t pass this cell')
                    label_stat1 = False
                else:
                    plot = ax.scatter(list1[d][0], list1[d][1], marker='o', s=60, lw=0, color = 'red')
            else:
                if label_stat2:
                    plot = ax.scatter(list1[d][0], list1[d][1], marker='o', s=60, lw=0, color = 'Black',
                                      label = 'snow course passes this cell')
                    label_stat2 = False
                else:
                    plot = ax.scatter(list1[d][0], list1[d][1], marker='o', s=60, lw=0, color = 'Black')

        plt.plot(path_x_np[0], path_y_np[0], lw = 4, color = 'green', marker='_',  alpha = 0.5,
                    label =f'Snow course ({snow_course_length} Meters)' )
        plt.scatter(path_x_np, path_y_np, marker='.', s=100, lw=0, color = 'green', alpha = 0.03)

        if draw_all_cells == 'yes':
            draw_all_cells = -1
        else:
            draw_all_cells = 0

        temp_x, temp_y = [],[]

        label_stat3 = True

        for i in range(cells_x_np2.size):
            temp_x.append(list1[i][0]+500)
            temp_x.append(list1[i][0]+500)
            temp_x.append(list1[i][0]-500)
            temp_x.append(list1[i][0]-500)
            temp_x.append(list1[i][0]+500)
            temp_y.append(list1[i][1]+500)
            temp_y.append(list1[i][1]-500)
            temp_y.append(list1[i][1]-500)
            temp_y.append(list1[i][1]+500)
            temp_y.append(list1[i][1]+500)

            if list1[i][3] > draw_all_cells:
                if label_stat3:
                    plt.plot(temp_x, temp_y, lw = 4, color = 'black', alpha = 0.2, label = "Confine a cell")
                    label_stat3 = False
                else:
                    plt.plot(temp_x, temp_y, lw = 4, color = 'black', alpha = 0.2, label = "")

            temp_x, temp_y = [],[]    

            for j in range(len(list1)):
                if cells_x_np2[i] == list1[j][0]:
                    if cells_y_np2[i] == list1[j][1]:

                        strcell = f'Cell No.: {list1[j][2]}'
                        if list1[j][3] == 0 : strcell = ""
                        plt.annotate(strcell, xy =(cells_x_np2[i]-450, cells_y_np2[i]+410), fontsize = 12,
                                     color = "blue")
                        if list1[j][3] == 0:
                            continue
                        else:
                            if list2[j][10] == 'North':
                                xx = 0
                                yy = 350
                            elif list2[j][10] == 'South':
                                xx = 0
                                yy = -350
                            elif list2[j][10] == 'North-East':
                                xx = 250
                                yy = 250
                            elif list2[j][10] == 'Easth':
                                xx = 350
                                yy = 0
                            elif list2[j][10] == 'South-East':
                                xx = 250
                                yy = -250                    
                            elif list2[j][10] == 'North-West':
                                xx = -250
                                yy = 250                    
                            elif list2[j][10] == 'West':
                                xx = -350
                                yy = 0

                            elif list2[j][10] == 'South-West':
                                xx = -250
                                yy = -250                        

                            plt.annotate("", xy =(cells_x_np2[i]+xx, cells_y_np2[i] + yy), fontsize = 12, 
                                     arrowprops = dict(facecolor = 'olive', width =4, alpha = 0.6),
                                     xytext=(cells_x_np2[i], cells_y_np2[i]),)                

                        strcell = f'EL.grad. {list2[j][9]}%'
                        if list1[j][3] == 0 : strcell = ""
                        plt.annotate(strcell, xy =(cells_x_np2[i]-450, cells_y_np2[i]+70), fontsize = 12,
                                     color = "olive")                

                        strnumber = f'points: {list1[j][3]}'
                        if list1[j][3] == 0 : strnumber = ""  
                        plt.annotate(strnumber, xy =(cells_x_np2[i]+20, cells_y_np2[i]+410), fontsize = 12,
                                     color = 'darkgreen')

                        strnumber = f'{list2[j][10]}'
                        if list1[j][3] == 0 : strnumber = ""  
                        plt.annotate(strnumber, xy =(cells_x_np2[i]+30, cells_y_np2[i]+70), fontsize = 12,
                                     color = 'olive')               

                        str_average_swe = f'Average SWE: {int(list1[j][4])} (mm)'
                        if list1[j][3] == 0 : str_average_swe = ""
                        plt.annotate(str_average_swe, xy =(cells_x_np2[i]-450, cells_y_np2[i]-260), fontsize = 12,
                                     color = 'green')                

                        str_std = f'  {int(list1[j][5])}, '
                        if list1[j][3] == 0 : str_std = ""
                        plt.annotate(str_std, xy =(cells_x_np2[i]-120, cells_y_np2[i]+200), fontsize = 12,
                                     color = 'black')       

                        str_min = f'    {int(list1[j][6])}]'
                        if list1[j][3] == 0 : str_min = ""
                        plt.annotate(str_min, xy =(cells_x_np2[i]+200, cells_y_np2[i]+200), fontsize = 12,
                                     color = 'black')

                        str_max = f'[{int(list1[j][7])}, '
                        if list1[j][3] == 0 : str_max = ""
                        plt.annotate(str_max, xy =(cells_x_np2[i]-450, cells_y_np2[i]+200), fontsize = 12,
                                     color = 'black')

                        str_maxminstd1 = f'[Max.         Std.          Min.]'
                        if list1[j][3] == 0 : str_maxminstd1 = ""
                        plt.annotate(str_maxminstd1, xy =(cells_x_np2[i]-450, cells_y_np2[i]+310), fontsize = 12,
                                     color = 'black')                

                        obs = float(list1[j][4])
                        sim = float(list2[j][11])
                        accuracy = int((1-abs((obs-sim)/obs))*100)
                        str_accuracy = f'Accuracy: {accuracy} %'
                        if list1[j][3] == 0 : str_accuracy = ""
                        plt.annotate(str_accuracy, xy =(cells_x_np2[i]-450, cells_y_np2[i]-400), fontsize = 12,
                                     color = 'black')

                        model_swe = round(float(list2[j][11]),2)
                        str_max = f'Model SWE: {int(model_swe)} (mm)'
                        if list1[j][3] == 0 : str_max = ""
                        plt.annotate(str_max, xy =(cells_x_np2[i]-450, cells_y_np2[i]-120), fontsize = 12,
                                     color = 'deeppink')                

        if int((max(cells_x_2) - min(cells_x_2))/220) < 12:           
            plt.title(f"Snow course over cells grid\n({date_get}) file: {file_name}", fontsize = font_size + 6)
        else:
            plt.title(f"Snow course over cells grid ({date_get}) file: {file_name}", fontsize = font_size + 6)
        plt.xlabel('X coordinade')
        plt.ylabel('Y coordinade')

        ax.legend(loc=0, fontsize = 14)

        plt.savefig(f"{snowcourse2}_{snowcourse1}.png")

        plt.show()            

        list3 = deepcopy(list2)
        for i in range(len(list2)):
            if float(list2[i][4]) == 0:
                accuracy = 0
            else:
                obs = float(list2[i][4])
                sim = float(list2[i][11])
                accuracy = int((1-abs((obs-sim)/obs))*100)   
            list3[i].append(accuracy)

        accuracy = []
        sim_model = []
        for i in range(len(list3)):
            if list2[i][3]!=0:
                accuracy.append(list3[i][12])
                sim_model.append(list3[i][11])

        for i in range(len(cell_close_no)-len(accuracy)):
            accuracy.append(0)
            sim_model.append(0)

        snow_course_pd['SWE Model'] = sim_model    
        snow_course_pd['Accuracy'] = accuracy

        snow_course_pd.to_csv(f'{file_name}_cell_close_no.csv')                

        list_cell_no, list_swe_model  = [], []

        for item in range(len(list1)):
            if list1[item][3]== 0:
                continue
            list_cell_no.append(list1[item][2])

        for num in range(len(list_cell_no)):
            list_swe_model.append(all_swe_pd.loc[list_cell_no[num]+1][cell_swe_no])

        list_swe_model_flat = []
        for item in list_swe_model:
            item_str = str(item)
            list_swe_model_flat.append(int(item_str.split('.')[0]))
            
        fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize = (15,6))
        ax1.hist(list_swe_model_flat, color='y', alpha=0.3)
        ax1.set_xlabel(f"Snow Water Equivalent (mm) of passed cells\n{list_swe_model_flat}", fontsize=14)
        ax1.set_ylabel("frequency", fontsize=14)
        ax1.set_title(f"SWE Histogram of passed cells ({file_name})", fontsize = font_size + 0)

        ax2.hist(swe, bins=50,  color='r', alpha=0.3)
        ax2.set_xlabel(f"Snow Water Equivalent (mm) of snow course\nMin: {int(swe_np.min())}         Mean: {int(swe_np.mean())}         Max: {int(swe_np.max())}", fontsize=14)
        ax2.set_ylabel("frequency", fontsize=14)
        ax2.set_title(f"SWE Histogram of snow course ({file_name}", fontsize = font_size + 0)

        plt.savefig(f"{file_name}_3.png")

        plt.show()
            
print('_It is done_'*7)

# make a sound to notify that it is done
import winsound
for i in range(1400,3500,100):
    winsound.Beep(i, int(200*1500/i))
