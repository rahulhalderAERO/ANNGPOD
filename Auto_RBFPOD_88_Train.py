# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:03:09 2022

@author: rahul
"""

# Import Python Packages

import numpy as np
# from IPython import get_ipython
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
# get_ipython().run_line_magic('matplotlib', 'inline')

# Input Prameter for training 

index = 37 # which index you consider as a validation points (There are total of 88 parameters, our algorithm choose 22nd column)


train_size = [75] 

# Size of the approximate training points that needs to be deducted




list_plane = [0.96615,0.77795,1.59975,1.24475,1.56115,1.38115,1.07215,0.95944,0.14725,"0.14725m", 0.0]
             # planes of interest (combines both z plane (first 8) and x planes (last 3))
            



# Call Output Data

snapshots = np.load('Temp_Tabular/T_Highfidelity.npy') # Output Tabulated 
#matrix of Temperature in Kelvin associated with 11 planes

snapshots = snapshots.transpose() 
# The matrix size of  snapshots is (88*1443297)
# Size 88 is the total parametric points 
# 3 is the type of the parameters

# ====================Input parameter==========================================

# Tevap = [-15, -7.9, -3.25 , 4] # evaporator temperature variation (in degree C)
# Tamb = [16,32] # (in degree C)
# Vf_max = 100 # Max Fan velocity in percentages # (in %)
# Vf_min = 0 # Min Fan velocity in percentages # (in %) 
# Vf_step = 10 # Fan velocity step # (in %)

params0 = np.load('Temp_Tabular/params0.npy') # Input Tabulated matrix 
#['Tevap','Tamb','Vf_percentage']






# ====================Input and Output Validation Points=======================

Para_test_fixed = params0[index,:] # validation parameter input
Snapshot_test_fixed = snapshots[index,:] # validation parameter output


Para_train = np.delete(params0,index,axis = 0) # delete first the validation 
                                               # point from input training matrix
Snapshot_train = np.delete(snapshots,index,axis = 0) # delete first the 
                                                     # validation point from output training matrix

# Store the values in a new variables adding final at the end

Snapshot_train_final = Snapshot_train
Para_train_final = Para_train



# Save the validation points for future prediction

np.save("Para_test_fixed.npy",Para_test_fixed) # Input Parameter for Validation for future use



np.save("Snapshot_test_fixed.npy",Snapshot_test_fixed) # Output Parameter for Validation for future use


# Training Phase

for k in range(len(train_size)):
    
    
  list_array = np.load("Percentage_Of_Input _Mod/Traindata_{}.npy".format(train_size[0])) # input the random variations of numbers which need to be deducted.

  
  # list_array are those randomly generated numbers which row number will be subtracted from the main input matrix 
  # It is generated using the following code
  # list=[]
  # for i in range(train_size):
  #      r=random.randint(1,88-1) # 88 being the total parametric points.
  #      if r not in list: list.append(r)
  # list_array = np.array(list) 


  list_array_mod = np.sort(list_array, axis= None) # sort the random numbers 
  
  # Get sparse trained dictionery 
    
  for i in range(len(list_array_mod)):
         Snapshot_train_final = np.delete(Snapshot_train_final,list_array_mod[i]-i,axis = 0)
         Para_train_final = np.delete(Para_train_final,list_array_mod[i]-i,axis = 0)

  
 # Create the Database and Training Model


  db = Database(Para_train_final, Snapshot_train_final)
  
 # Generation of the POD coeffcients 
  
  pod = POD('svd', rank = -1 )
  rbf = RBF()
  rom = ROM(db, pod, rbf)
  rom.fit();
  eigs0=pod.singular_values
  modes0=pod.modes
  energy0 = np.cumsum(eigs0)/sum(eigs0)
  coeff0 = pod.reduce(db.snapshots.T)
  engLevel = 0.997 # Cumulative energy that you want to contain (Modify if you want to contain more energy)
  ind = np.argmax(energy0 > engLevel)  # ind of eigvalue greater than i
  r = ind +1    # index starts from 0
  pod = POD('svd', rank = int(r+1))
  rbf = RBF()
  rom = ROM(db, pod, rbf)
  rom.fit();
  eigs=pod.singular_values
  modes=pod.modes
  
  
  # Save the Trained Model
  
  rom.save('ezyrb_RBFPOD_{}.rom'.format(train_size[k]))
  

  












