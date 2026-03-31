# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:50:47 2023

@author: rahul
"""

#=======================Input packages =======================================

import numpy as np
# from IPython import get_ipython
from ezyrb import ReducedOrderModel as ROM
# get_ipython().run_line_magic('matplotlib', 'inline')
from smithers.io import VTUHandler as Handler
from scipy.io import savemat,loadmat
import os
from matplotlib import pyplot as plt




# define a function for calculating mean absolute error

def mae(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.abs(np.subtract(actual,pred)).mean()




#======================Input Parameter ========================================


# Input parameter

index = 37 # which index you consider as a validation points

# Corresponding Experimental Data Provided by Electrolux

Exp_Index = 8

train_size = [75] # Size of the training points that needs to be deducted (It should be same as indicated in the training part)

list_plane = [0.96615,0.77795,1.59975,1.24475,1.56115,1.38115,1.07215,0.95944,0.14725,"0.14725m", 0.0]
             # planes of interest (combines both z plane and x planes)

Size_Mat = np.load('Temp_Tabular/Size_Mat.npy') # Size of different grid size associated with different planes 


# Prediction at different size of the sparse training points

for k in range(len(train_size)): 

  mae_error =[]
  
  #load the ROM model saved in the Training Phase
  
  rom = ROM.load('ezyrb_RBFPOD_{}.rom'.format(train_size[k]))
  

  # Online phase : Predict form the new parameter
    
  new_mu = np.load('Para_test_fixed.npy') # Call the validation parameter ["T_{evaporator} (degree C) T_{ambient} (degree C) Fan_Velocity (%) "]
  pred_sol = rom.predict(new_mu)
  pred_sol = pred_sol.T
  
  #  Benchmark CFD data
  Snapshot_test_fixed = np.load('Snapshot_test_fixed.npy').T
  
  # Post-Processing step 1 : Different ranges of the gridpoints associated with the planes  
  for case in range(1,12):
      range_a = 0 
      for i in range(case):
          range_a = range_a + Size_Mat[i]
          range_b = range_a - Size_Mat[case-1]

      # error associated with different planes
      err = mae(Snapshot_test_fixed[range_b:range_a],pred_sol[range_b:range_a])
    
      print("mae error is =", err )
     
      #post processing : Generate vtu files from python output
     
     
      test_file = "File_Vtk/Temp@{}_All.vtu".format(list_plane[case-1]) # required for mesh information
      data = Handler.read(test_file)
      Points = data["points"]
      data_recons_PODRBF = data
      data_error = data
     
      # Replace with absolute error computed at grid points 
     
      data_recons_PODRBF["point_data"]["T"]= abs(Snapshot_test_fixed[range_b:range_a]-pred_sol[range_b:range_a])
      dir = 'RBFPOD_Error_88_Experiment/RBFPOD_Error_{}_{}_997'.format(index,train_size[k])
      if not os.path.exists(dir):
            os.makedirs(dir)
    
    
      Handler.write("RBFPOD_Error_88_Experiment/RBFPOD_Error_{}_{}_997/T_Error_{}_RBFPOD.vtu".format(index,train_size[k],list_plane[case-1]),data_recons_PODRBF)
    
      mae_error.append(err)


  # Input Array of all the errors
  
  
  mae_array = np.array(mae_error).reshape(-1,1)

  mdic = {"RBF_mae_array_{}_{}".format(index,train_size[k]) : mae_array}
           
  savemat("RBFPOD_Error_88_Experiment/RBF_mae_array_{}_{}_997.mat".format(index,train_size[k]), mdic)
  
  
  
  
  #----------------------------------------------------------------------------
  
  # At the 26 sensor points 
  
  #----------------------------------------------------------------------------
  
  
  #Specify the 26 sensor locations which we will validate now with ANNGPOD

  sensor_x_test = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_x_test.npy')

  # [0,0.17925,0.17925,-0.17925,-0.17925,
  #                   0,0.17925,0.17925,-0.17925,-0.17925,
  #                   0,0.17925,0.17925,-0.17925,-0.17925,
  #                   -0.00136,0.14725,0.14725,-0.14996,-0.14996,
  #                   0.12805,0,-0.12805,
  #                   -0.0715,-0.0715,-0.00012]

  sensor_y_test = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_y_test.npy')

  # [-0.26173,-0.15226,-0.3712,-0.15226,-0.3712,
  #                   -0.26173,-0.15226,-0.3712,-0.15226,-0.3712,
  #                   -0.26173,-0.15226,-0.3712,-0.15226,-0.3712,                 
  #                   -0.22044,-0.35845,-0.08244,-0.08244,-0.35845,
  #                   -0.27622,-0.27576,-0.27622,
  #                   -0.0578,-0.0578,-0.0595]

  sensor_z_test = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_z_test.npy')

  # [1.56115,1.56115,1.56115,1.56115,1.56115,
  #                   1.38115,1.38115,1.38115,1.38115,1.38115,
  #                   1.07215,1.07215,1.07215,1.07215,1.07215,                 
  #                   0.77795,0.77795,0.77795,0.77795,0.77795,
  #                   0.95944,0.95944,0.95944,
  #                   1.59975,1.24475,0.96615]

  eps = 0.00001
  wt_z = 100000
  wt_x = 1
  wt_y = 1

  x_mat = np.load('Temp_Tabular/x_mat_all_plane.npy')
  y_mat = np.load('Temp_Tabular/y_mat_all_plane.npy')
  z_mat = np.load('Temp_Tabular/z_mat_all_plane.npy')

  def closest_value(input_list_x, input_list_y, input_list_z, input_value_x, input_value_y, input_value_z):
      
      
      i = (wt_x*(np.square(input_list_x - input_value_x))/np.square(input_value_x+eps)
      +wt_y*(np.square(input_list_y - input_value_y))/np.square(input_value_y+eps)+
      wt_z*(np.square(input_list_z - input_value_z))/np.square(input_value_z+eps)).argmin()
      points = [input_list_x[i],input_list_y[i],input_list_z[i],i]
      
      return points 

  val_loc_test = np.zeros((len(sensor_x_test),4))

  for i in range(len(sensor_x_test)):  
      val_loc_test[i,:] = closest_value(x_mat[:,0],y_mat[:,0],z_mat[:,0],sensor_x_test[i],sensor_y_test[i],sensor_z_test[i])

    

  # Comparison of the CFD, ROM and Expeiment

  list_sensor_test = val_loc_test[:,3].astype(int)
  snapshots_26_CFD = Snapshot_test_fixed[list_sensor_test].reshape(-1,1)
  snapshots_26_Exp = loadmat("Exp_Num_Comparison.mat")['Experiment'][:,Exp_Index].reshape(-1,1)
  snapshots_26_ROM = pred_sol[list_sensor_test].reshape(-1,1)


  dir = 'RBFPOD_Error_88_Experiment'
  if not os.path.exists(dir):
    os.makedirs(dir)
    
    
  # Plot CFD, RBFPOD ROM, ANNGPOD ROM and compare with Experiments. 
 
  # plt.plot(np.arange(0,len(CFD_Aggregated)),CFD_Aggregated[:,0] , label ='CFD_Aggregated')
  plt.plot(np.arange(0,len(snapshots_26_Exp)),snapshots_26_Exp[:,0] ,'-o', label ='Experiment')
  plt.plot(np.arange(0,len(snapshots_26_CFD)),snapshots_26_CFD[:,0] ,'-o', label='CFD')
  plt.plot(np.arange(0,len(snapshots_26_ROM)),snapshots_26_ROM[:,0] ,'-o', label ='RBFPOD ROM')

  # plt.legend(['Experiment','CFD from snap','ROM'])
  plt.xlabel('Sensor Locs.',fontsize=14)
  plt.ylabel('T (K)', fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)


  # # legend settings
  plt.ylim(273, 290)
  # plt.xlim(0.5, 27.5)
   

  # plt.yscale('log')
  plt.legend(ncol=1, loc=2, fontsize=15) # 9 means top center
  plt.savefig('RBFPOD_Error_88_Experiment/plot_ExpCFDROM_{}_{}_{}.pdf'.format(index,Exp_Index,train_size[0]), dpi=300)
  plt.show()

  # Compute Error Between the CFD ROMs and Experiments

  Diff_CFD_ROM = abs(snapshots_26_CFD-snapshots_26_ROM)
  Diff_CFD_Exp = abs(snapshots_26_Exp-snapshots_26_CFD)
  Diff_Exp_ROM = abs(snapshots_26_Exp-snapshots_26_ROM)


  mdic_error = {"Diff_CFD_ROM" : Diff_CFD_ROM,"Diff_CFD_Exp" : Diff_CFD_Exp, "Diff_Exp_ROM" : Diff_Exp_ROM}             
  mdic_Actual = {"snapshots_26_Exp" : snapshots_26_Exp,"snapshots_26_CFD" : snapshots_26_CFD, "snapshots_26_ROM" : snapshots_26_ROM}


  savemat("RBFPOD_Error_88_Experiment/RBFPOD_ExpCFDROM_error_{}_{}_26sensors.mat".format(index,train_size[k]), mdic_error)
  savemat("RBFPOD_Error_88_Experiment/RBFPOD_ExpCFDROM_actual_{}_{}_26sensors.mat".format(index,train_size[k]), mdic_Actual)


# #------------------------------------------------------------------------------
# #-------At 5 sensor points ----------------------------------------------------  
# #------------------------------------------------------------------------------





#Specify the 26 sensor locations which we will validate now with ANNGPOD

sensor_x = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_x.npy')

# [0,0.17925,0.17925,-0.17925,-0.17925,
#                   0,0.17925,0.17925,-0.17925,-0.17925,
#                   0,0.17925,0.17925,-0.17925,-0.17925,
#                   -0.00136,0.14725,0.14725,-0.14996,-0.14996,
#                   0.12805,0,-0.12805,
#                   -0.0715,-0.0715,-0.00012]

sensor_y = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_y.npy')

# [-0.26173,-0.15226,-0.3712,-0.15226,-0.3712,
#                   -0.26173,-0.15226,-0.3712,-0.15226,-0.3712,
#                   -0.26173,-0.15226,-0.3712,-0.15226,-0.3712,                 
#                   -0.22044,-0.35845,-0.08244,-0.08244,-0.35845,
#                   -0.27622,-0.27576,-0.27622,
#                   -0.0578,-0.0578,-0.0595]

sensor_z = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_z.npy')

# [1.56115,1.56115,1.56115,1.56115,1.56115,
#                   1.38115,1.38115,1.38115,1.38115,1.38115,
#                   1.07215,1.07215,1.07215,1.07215,1.07215,                 
#                   0.77795,0.77795,0.77795,0.77795,0.77795,
#                   0.95944,0.95944,0.95944,
#                   1.59975,1.24475,0.96615]

eps = 0.00001
wt_z = 100000
wt_x = 1
wt_y = 1

x_mat = np.load('Temp_Tabular/x_mat_all_plane.npy')
y_mat = np.load('Temp_Tabular/y_mat_all_plane.npy')
z_mat = np.load('Temp_Tabular/z_mat_all_plane.npy')

def closest_value(input_list_x, input_list_y, input_list_z, input_value_x, input_value_y, input_value_z):
    
    
    i = (wt_x*(np.square(input_list_x - input_value_x))/np.square(input_value_x+eps)
    +wt_y*(np.square(input_list_y - input_value_y))/np.square(input_value_y+eps)+
    wt_z*(np.square(input_list_z - input_value_z))/np.square(input_value_z+eps)).argmin()
    points = [input_list_x[i],input_list_y[i],input_list_z[i],i]
    
    return points 

val_loc_test_5 = np.zeros((len(sensor_x),4))
val_loc_5_from_26 = np.zeros((len(sensor_x),4))

for i in range(len(sensor_x)):  
    val_loc_test_5[i,:] = closest_value(x_mat[:,0],y_mat[:,0],z_mat[:,0],sensor_x[i],sensor_y[i],sensor_z[i])

for i in range(len(sensor_x)):  
    val_loc_5_from_26[i,:] = closest_value(sensor_x_test,sensor_y_test,sensor_z_test,sensor_x[i],sensor_y[i],sensor_z[i])
 

# Comparison of the CFD, ROM and Expeiment

list_sensor_test = val_loc_test_5[:,3].astype(int)
list_sensor_test_5 = val_loc_5_from_26[:,3].astype(int)

snapshots_5_CFD = Snapshot_test_fixed[list_sensor_test].reshape(-1,1)
snapshots_5_Exp = loadmat("Exp_Num_Comparison.mat")['Experiment'][:,Exp_Index].reshape(-1,1)[list_sensor_test_5].reshape(-1,1)
snapshots_5_ROM = pred_sol[list_sensor_test].reshape(-1,1)


dir = 'RBFPOD_Error_88_Experiment'
if not os.path.exists(dir):
  os.makedirs(dir)
  
  
# Plot CFD, RBFPOD ROM, ANNGPOD ROM and compare with Experiments. 

# plt.plot(np.arange(0,len(CFD_Aggregated)),CFD_Aggregated[:,0] , label ='CFD_Aggregated')
plt.plot(np.arange(0,len(snapshots_5_Exp)),snapshots_5_Exp[:,0] ,'-o', label ='Experiment')
plt.plot(np.arange(0,len(snapshots_5_CFD)),snapshots_5_CFD[:,0] ,'-o', label='CFD')
plt.plot(np.arange(0,len(snapshots_5_ROM)),snapshots_5_ROM[:,0] ,'-o', label ='RBFPOD ROM')

# plt.legend(['Experiment','CFD from snap','ROM'])
plt.xlabel('Sensor Locs.',fontsize=14)
plt.ylabel('T (K)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# # legend settings
plt.ylim(273, 290)
# plt.xlim(0.5, 27.5)

# plt.yscale('log')
plt.legend(ncol=1, loc=2, fontsize=15) # 9 means top center
plt.savefig('RBFPOD_Error_88_Experiment/plot_ExpCFDROM_{}_{}_{}_5_points.pdf'.format(index,Exp_Index,train_size[0]), dpi=300)
plt.show()


# Compute Error Between the CFD ROMs and Experiments

Diff_CFD_ROM = abs(snapshots_5_CFD-snapshots_5_ROM)
Diff_CFD_Exp = abs(snapshots_5_Exp-snapshots_5_CFD)
Diff_Exp_ROM = abs(snapshots_5_Exp-snapshots_5_ROM)

T_strat_ROM = max(abs(snapshots_5_ROM[0,0]-snapshots_5_ROM[1,0]),abs(snapshots_5_ROM[1,0]-snapshots_5_ROM[2,0]),abs(snapshots_5_ROM[2,0]-snapshots_5_ROM[0,0]))
T_strat_CFD = max(abs(snapshots_5_CFD[0,0]-snapshots_5_CFD[1,0]),abs(snapshots_5_CFD[1,0]-snapshots_5_CFD[2,0]),abs(snapshots_5_CFD[2,0]-snapshots_5_CFD[0,0]))
T_strat_EXP = max(abs(snapshots_5_Exp[0,0]-snapshots_5_Exp[1,0]),abs(snapshots_5_Exp[1,0]-snapshots_5_Exp[2,0]),abs(snapshots_5_Exp[2,0]-snapshots_5_Exp[0,0]))

print("T_strat_ROM is ====", T_strat_ROM)
print("T_strat_CFD is ====", T_strat_CFD)
print("T_strat_EXP is ====", T_strat_EXP)




mdic_error = {"Diff_CFD_ROM" : Diff_CFD_ROM,"Diff_CFD_Exp" : Diff_CFD_Exp, "Diff_Exp_ROM" : Diff_Exp_ROM}             
mdic_Actual = {"snapshots_5_Exp" : snapshots_5_Exp,"snapshots_5_CFD" : snapshots_5_CFD, "snapshots_5_ROM" : snapshots_5_ROM}


savemat("RBFPOD_Error_88_Experiment/RBFPOD_ExpCFDROM_error_{}_{}_5sensors.mat".format(index,train_size[k]), mdic_error)
savemat("RBFPOD_Error_88_Experiment/RBFPOD_ExpCFDROM_actual_{}_{}_5sensors.mat".format(index,train_size[k]), mdic_Actual)























