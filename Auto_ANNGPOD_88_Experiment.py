# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:03:09 2022

@author: rahul
"""


# Import Python Packages


import numpy as np
from IPython import get_ipython
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from Feed_Forward import FeedForward
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from smithers.io import VTUHandler as Handler
import os
from scipy.io import savemat,loadmat
from matplotlib import pyplot as plt




# Input Prameter for training 

# which index you consider as a validation points (There are total of 88 parameters, our algorithm choose 22nd column)
# varies between 0 --> 87

index = 37  

# Corresponding Experimental Data Provided by Electrolux

Exp_Index = 8

# Size of the approximate training points that needs to be deducted. 
# Size of the training dataset

train_size = [75]


# List of planes used to name output files. The length of the list should match the number of planes in the training dataset
list_plane = [0.96615,0.77795,1.59975,1.24475,1.56115,1.38115,1.07215,0.95944,0.14725,"0.14725m", 0.0]
             # planes of interest (combines both z plane (first 8) and x planes (last 3))


#------------------------------------------------------------------------------
# Input from USER
#------------------------------------------------------------------------------

# Call Output Data

# Load the organized, aggregated temperature data (matrix side of snapshots = 88 x 1443297)
# Size 88 is the total parametric points (i.e., 88 combinations of parameters / CFD runs)
# Size 1443297 is the total number of temperatures points (i.e., grid points) accross all planes
# Matrix of Temperature in Kelvin associated with 11 planes

snapshots = np.load('Temp_Tabular/T_Highfidelity.npy')
snapshots = snapshots.transpose()

# ====================Input parameter==========================================

# Size of dataset is [ 4 x 2 x 11 = 88 ]

# 4 Evaporator Temperatuers
#     Tevap = [-15, -7.9, -3.25 , 4] # evaporator temperature variation (in degree C)

# 2 Ambient Temperatures:
#     Tamb = [16, 32] # (in degree C)

# 11 fan velocities distributed as:
#     Vf_max = 100 # Max Fan velocity in percentages # (in %)
#     Vf_min = 0 # Min Fan velocity in percentages # (in %) 
#     Vf_step = 10 # Fan velocity step # (in %)

params0 = np.load('Temp_Tabular/params0.npy') # Input Tabulated matrix 

# Dataset input matrix is structured as :
# ['Tevap','Tamb','Vf_percentage']

# ['Tevap_0', 'Tamb_0', 'Vf_percentage_0']
# [... ,       ...    ,  ...]
# ['Tevap_N', 'Tamb_N', 'Vf_percentage_N']

# Size of grid points associated with different planes 

Size_Mat = np.load('Temp_Tabular/Size_Mat.npy')


# Location of the x,y and z co-ordinate
# XYZ coordinates required to locate experimental sensor locations on CFD mesh


x_mat = np.load('Temp_Tabular/x_mat_all_plane.npy')
y_mat = np.load('Temp_Tabular/y_mat_all_plane.npy')
z_mat = np.load('Temp_Tabular/z_mat_all_plane.npy')

Temp_experimental =    loadmat("Exp_Num_Comparison.mat")['Experiment_5points'][:,Exp_Index].reshape(-1,1) + 273.15# Snapshot_test_fixed[list_sensor].reshape(-1,1)

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Algorithm STARTS here 
#------------------------------------------------------------------------------

# Create Function for the Mean Absolute Error

def mae(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.abs(np.subtract(actual,pred)).mean()

# Create Max and Min Value of a vector and normalize them.
# For a given vector, the max and min are computed and it is scaled between 0 and 1

def val_scale(vec):
    Size_y = vec.size(1)
    scaled_vec = torch.zeros(vec.size())
    max_val_vec = torch.zeros(Size_y)
    min_val_vec = torch.zeros(Size_y)
    for i in range(Size_y):
        max_val_vec[i], max_idxs = torch.max(vec[:,i], dim=0)
        min_val_vec[i], min_idxs = torch.min(vec[:,i], dim=0)
        scaled_vec[:,i] =  (vec[:,i]-min_val_vec[i])/(max_val_vec[i]-min_val_vec[i])
    return scaled_vec,max_val_vec,min_val_vec

# Normalize a vector with a given Max and Min Value. 
# Same conept as above, but we provide the scale for normalization via max_val and min_val

def scale_with_given_value(vec,max_val,min_val):
    Size_y = vec.size(1)
    scaled_vec1 = torch.zeros(vec.size()) 
    for i in range(Size_y):
        scaled_vec1[:,i] =  (vec[:,i] - min_val[i])/(max_val[i]-min_val[i])
    return scaled_vec1 
 
# Rescale to original value from a normalized vector with given max and minimum value
       
def rescale_to_original(vec,max_val,min_val):
    Size_y = vec.size(1)
    rescaled_vec = torch.zeros(vec.size()) 
    for i in range(Size_y):
        rescaled_vec[:,i] =  (vec[:,i])*(max_val[i]-min_val[i]) +  min_val[i]*torch.ones(vec.size(0))
    return rescaled_vec


# Define a function which will identify the narest possible nodes.

# Define a function which will identify the nearest possible nodes.
def closest_value(input_list_x, input_list_y, input_list_z, input_value_x, input_value_y, input_value_z, eps=0.00001, wt=100):

    # e.g., input_list_[x,y,z] is the vector of all CFD grid points
    # e.g., input_value_[x,y,z] is the point for which we want to find the nearest grid point
    # eps parameter used to avoid division by 0
    # wt parameter increases the importance of z so that the point is located on the correct plane

    i = ((np.square(input_list_x - input_value_x))/np.square(input_value_x+eps)
    +(np.square(input_list_y - input_value_y))/np.square(input_value_y+eps)+
    wt*(np.square(input_list_z - input_value_z))/np.square(input_value_z+eps)).argmin()

    points = [input_list_x[i],input_list_y[i],input_list_z[i],i]

    return points

# Hyperparameter associated with the learning rate (step decay): 
# Learning rate is an input for the neural network and varies as the optimization advances

def step_decay(epoch):
    if epoch < 5000:
        lr = 0.01
    elif epoch<10000:
        lr = 0.005
    else:
        lr = 1e-3
    return lr


#--------------------------------------------------------------------------            
# Training and Test Separation from the parameter space 
#--------------------------------------------------------------------------

Para_test_fixed = params0[index,:] # Test parameter
Snapshot_test_fixed = snapshots[index,:] # Test Snapshots

Para_train = np.delete(params0,index,axis = 0)  # Train parameter
Snapshot_train = np.delete(snapshots,index,axis = 0) # Train Snapshots 

# Save the Train parameter and Snapshots in a new variables
Snapshot_train_final = Snapshot_train
Para_train_final = Para_train



for k in range(len(train_size)): 
       
    
    
    list_array = np.load("Percentage_Of_Input _Mod/Traindata_{}.npy".format(train_size[k])) # input the random variations of numbers which need to be deducted.

    
    # list_array are those randomly generated numbers which row number will be subtracted from the main input matrix 
    # It is generated using the following code
    # list=[]
    # for i in range(train_size):
    #      r=random.randint(1,88-1) # 88 being the total parametric points.
    #      if r not in list: list.append(r)
    # list_array = np.array(list) 


    list_array_mod = np.sort(list_array, axis= None) # sort the random numbers 
    
    # Get sparse trained dictionery by removing random points indices
      
    for i in range(len(list_array_mod)):
            Snapshot_train_final = np.delete(Snapshot_train_final,list_array_mod[i]-i,axis = 0)
            Para_train_final = np.delete(Para_train_final,list_array_mod[i]-i,axis = 0)
           
    #-------------------------------------------------------------------------- 
    # Generate the POD coeffcients
    #--------------------------------------------------------------------------

    # Create the Database and Training Model
    
    db = Database(Para_train_final, Snapshot_train_final)
    
    
    # Generation of the POD coeffcients 

    # engLevel: Cumulative energy that you want to contain (Modify if you want to contain more energy)
    # Energy level is varied to minimize the projection error using POD.
    # The higher the energy level, less is the projection error, but the number of modes of the POD increases, which in turn induces interpolation error 
    # Therefore, final check must be done based on the total error at validation point
    # Rule of thumb: Minimum value should be 0.99. Optimal value in this case tuned to 0.997 by verifying validation point error
    engLevel = 0.997
    
    
    # Initialize pod and rbf 
    
    # RBF class is called to initialise other functions # RBF is not used in the present work
    
    
    pod = POD('svd', rank = -1 )
    rbf = RBF()

    # Use the ROM function to calculate all possible modes and associated energy
    # Store all modes energy levels in energy0
    rom = ROM(db, pod, rbf)
    rom.fit();
    eigs0=pod.singular_values
    modes0=pod.modes
    energy0 = np.cumsum(eigs0)/sum(eigs0)
    coeff0 = pod.reduce(db.snapshots.T)

    # Get mode indices where the energy is high than minimum engLevel
    ind = np.argmax(energy0 > engLevel)  # ind of eigvalue greater than i

    # To be safe, add one index 
    r = ind +1    # index starts from 0

    # Compute the final modes associated with minimum energy level and fit final ROM
    pod = POD('svd', rank = int(r+1))
    rbf = RBF()
    rom = ROM(db, pod, rbf)
    rom.fit();
    eigs=pod.singular_values
    modes=pod.modes
    coeff = np.matmul(modes.T,Snapshot_train_final.T)       
    
    
    #--------------------------------------------------------------------------
    # Specify the Sensors' Location
    #--------------------------------------------------------------------------

    # Sensor locations where experimental data is provided 
    sensor_x = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_x.npy')#[0.17925,0.17925,0.17925,0,0.14725]
    sensor_y = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_y.npy')
    sensor_z = np.load('Inputs_to_ANNGPOD/Sensor_Locations/sensor_z.npy')
    


    # Initialize array of sensor locations
    val_loc = np.zeros((len(sensor_x),4)) 
    
    # Find out where the sensors are placed co-ordinate wise on the mesh
    for i in range(len(sensor_x)):  
        val_loc[i,:] = closest_value(x_mat[:,0],y_mat[:,0],z_mat[:,0],sensor_x[i],sensor_y[i],sensor_z[i])
    
    # List of grid point indices corresponding to each sensor (location in the temperature solution vector)
    list_sensor = val_loc[:,3].astype(int)
    
    # Truncate the modes corresponding to the modes_truncated
    # Reduce modes array to modes corresponding to sensor locations
    modes_truncated = modes[list_sensor, :]
    
    # Sparse dataset available at new parameter in the index
    # In this example we assume sensor temperature is equal to CFD temperature @ sensor location
    # In next step, Temp_experimental is provided as an input alongside sensor_[x,y,z]

    
    # Sparse dataset available at known parameters 
    Temp_Numerical = Snapshot_train_final[:,list_sensor].T

    # Concatenate these two datasets.
    Temp_Interest = torch.tensor(np.concatenate((Temp_Numerical,Temp_experimental), axis = 1)).float()

    # Normalize the dataset
    Temp_Interest_scaled,max_temp_interest,min_temp_interest = val_scale(Temp_Interest)  

    # Generate the ANN model
    # Number of iterations to optimize the ANN
    num_epochs = 20000
    
    # Input and Output matrix in ANN

    # Concatenate training and test parameters in one input parameter matrix
    input_matrix = np.concatenate((Para_train_final, Para_test_fixed.reshape(-1,1).T), axis =0)

    # Coefficient matrix from POD based on training data only
    output_matrix = coeff.T
    
    # Convert them to torch tensor (Input and Output matrix in ANN)
    input_tensor = torch.from_numpy(input_matrix).float()
    output_tensor = torch.from_numpy(output_matrix).float()
    
    # Normalze the input and output matrix
    input_tensor_scaled,max_input,min_input = val_scale(input_tensor)
    output_tensor_scaled,max_output,min_output = val_scale(output_tensor)

    # Define input and output dimension and ANN model with hyper-parameters defined.    
    input_dim = Para_train_final.shape[1] 
    output_dim = modes.shape[1]

    # FeedForward is an ANN-based regression function
    # Class defined in Feed_Forward.py
    model = FeedForward(input_dim, output_dim)
    
    # Define the Loss critera : Here it is MSE (mean square error) loss
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.75)
    
    # Train the model
    num_epochs = num_epochs # Total number of iteration in the ANN 
    learning_rates = []
    for epoch in range(num_epochs):

        # Forward pass      
        y_pred = model(input_tensor_scaled)
        y_pred_data_avail = y_pred[:-1,:]
        
        # loss part from the POD-ANN part (to minimize)
        loss1 = criterion(y_pred_data_avail, output_tensor_scaled)
            
        # Get y_pred at the sesnor location
        y_pred_rescaled = rescale_to_original(y_pred,max_output,min_output)
        T_pred = torch.matmul(torch.tensor(modes_truncated).float(), y_pred_rescaled.T)
        T_pred_scaled = scale_with_given_value(T_pred,max_temp_interest,min_temp_interest)
        
        #Compute loss2 from the GPOD part 
        loss2 = criterion(T_pred_scaled.T, Temp_Interest_scaled.T)

        # Add both the loss term
        w1 = 1.
        w2 = 1.
        loss =  w1*loss1 + w2*loss2

        # Define Optimizer         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Define Learning Rate
        lr = scheduler.get_last_lr()[0]        
        
        # Print the loss every 1000 epochs
        if (epoch+1) % 100 == 0:
          print('Epoch %d loss: %.7f lr: %.4f' % (epoch+1, loss , lr))
        
    Rescaled_ypred =  (rescale_to_original(y_pred,max_output,min_output)).T.detach().numpy()
    Temp1 = np.matmul(modes,Rescaled_ypred)
    pred_sol = Temp1[:,-1]

# -----------------------------------------------------------------------------    
    # Mean Absolute Error associated with every plane and Absolute Error 
#------------------------------------------------------------------------------
    mae_error =[]
    
    for case in range(1,len(list_plane)+1):
    
        range_a = 0 
        for i in range(case):
            range_a = range_a + Size_Mat[i]
            range_b = range_a - Size_Mat[case-1]
        err = mae(Snapshot_test_fixed[range_b:range_a],pred_sol[range_b:range_a])
    
        print("mae error is =", err )
        test_file = "File_Vtk/Temp@{}_All.vtu".format(list_plane[case-1])
        data = Handler.read(test_file)
        Points = data["points"]
        data_recons_PODRBF = data
        data_pred = data
        
        # data_recons_PODRBF["point_data"]["T"]= abs(Snapshot_test_fixed[range_b:range_a]-pred_sol[range_b:range_a])
        data_pred["point_data"]["T"]= pred_sol[range_b:range_a]
        dir = 'ANNGPOD_Error_88_Experiment/ANNGPOD_Error_{}_{}'.format(index,train_size[k])
        if not os.path.exists(dir):
          os.makedirs(dir)
        # Handler.write("ANNGPOD_Error_88_w1_1_w2_10/ANNGPOD_Error_{}_{}/T_Error_{}_ANNGPOD.vtu".format(index,train_size[k],list_plane[case-1]),data_recons_PODRBF)    
        Handler.write("ANNGPOD_Error_88_Experiment/ANNGPOD_Error_{}_{}/T_pred_{}_ANNGPOD.vtu".format(index,train_size[k],list_plane[case-1]),data_pred)    
        mae_error.append(err)
    
    # Save the mea error and absolute error in the folder "ANNGPOD_Error_88"     
    mae_array = np.array(mae_error).reshape(-1,1)
    mdic = {"ANNGPOD_mae_array_{}_{}".format(index,train_size[k]) : mae_array}             
    savemat("ANNGPOD_Error_88_Experiment/ANNGPOD_mae_array_{}_{}.mat".format(index,train_size[k]), mdic)



#-----------------------------------------------------------------------------
#Specify the 26 sensor locations which we will validate now with ANNGPOD
#------------------------------------------------------------------------------

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
snapshots_26_RBFPOD = loadmat("RBFPOD_Error_88_Experiment/RBFPOD_ExpCFDROM_actual_{}_{}_26sensors.mat".format(index,train_size[0]))['snapshots_26_ROM'].reshape(-1,1)


dir = 'ANNGPOD_Error_88_Experiment'
if not os.path.exists(dir):
  os.makedirs(dir)
  
  
# Plot CFD, RBFPOD ROM, ANNGPOD ROM and compare with Experiments. 

# plt.plot(np.arange(0,len(CFD_Aggregated)),CFD_Aggregated[:,0] , label ='CFD_Aggregated')
plt.plot(np.arange(0,len(snapshots_26_Exp)),snapshots_26_Exp[:,0] ,'-o', label ='Experiment')
plt.plot(np.arange(0,len(snapshots_26_CFD)),snapshots_26_CFD[:,0] ,'-o', label='CFD')
plt.plot(np.arange(0,len(snapshots_26_ROM)),snapshots_26_ROM[:,0] ,'-o', label ='ANNGPOD ROM')
plt.plot(np.arange(0,len(snapshots_26_RBFPOD)),snapshots_26_RBFPOD[:,0] ,'-o', label ='RBFPOD ROM')

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
plt.savefig('ANNGPOD_Error_88_Experiment/plot_ExpCFDROM_{}_{}_{}.pdf'.format(index,Exp_Index,train_size[0]), dpi=300)
plt.show()

# Compute Error Between the CFD ROMs and Experiments

Diff_CFD_ROM = abs(snapshots_26_CFD-snapshots_26_ROM)
Diff_CFD_Exp = abs(snapshots_26_Exp-snapshots_26_CFD)
Diff_Exp_ROM = abs(snapshots_26_Exp-snapshots_26_ROM)


mdic_error = {"Diff_CFD_ROM" : Diff_CFD_ROM,"Diff_CFD_Exp" : Diff_CFD_Exp, "Diff_Exp_ROM" : Diff_Exp_ROM}             
mdic_Actual = {"snapshots_26_Exp" : snapshots_26_Exp,"snapshots_26_CFD" : snapshots_26_CFD, "snapshots_26_ROM" : snapshots_26_ROM}


savemat("ANNGPOD_Error_88_Experiment/ANNGPOD_ExpCFDROM_error_{}_{}_26sensors.mat".format(index,train_size[k]), mdic_error)
savemat("ANNGPOD_Error_88_Experiment/ANNGPOD_ExpCFDROM_actual_{}_{}_26sensors.mat".format(index,train_size[k]), mdic_Actual)







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
snapshots_5_ROM_ANNGPOD = pred_sol[list_sensor_test].reshape(-1,1)
snapshots_5_ROM_RBFPOD = snapshots_26_RBFPOD[list_sensor_test_5].reshape(-1,1)


dir = 'ANNGPOD_Error_88_Experiment'
if not os.path.exists(dir):
  os.makedirs(dir)
  
  
# Plot CFD, RBFPOD ROM, ANNGPOD ROM and compare with Experiments. 

# plt.plot(np.arange(0,len(CFD_Aggregated)),CFD_Aggregated[:,0] , label ='CFD_Aggregated')
plt.plot(np.arange(0,len(snapshots_5_Exp)),snapshots_5_Exp[:,0] ,'-o', label ='Experiment')
plt.plot(np.arange(0,len(snapshots_5_CFD)),snapshots_5_CFD[:,0] ,'-o', label='CFD')
plt.plot(np.arange(0,len(snapshots_5_ROM_ANNGPOD)),snapshots_5_ROM_ANNGPOD[:,0] ,'-o', label ='ANNGPOD ROM')
plt.plot(np.arange(0,len(snapshots_5_ROM_RBFPOD)),snapshots_5_ROM_RBFPOD[:,0] ,'-o', label ='RBFPOD ROM')

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
plt.savefig('ANNGPOD_Error_88_Experiment/plot_ExpCFDROM_{}_{}_{}_5_points.pdf'.format(index,Exp_Index,train_size[0]), dpi=300)
plt.show()


# Compute Error Between the CFD ROMs and Experiments

Diff_CFD_ROM = abs(snapshots_5_CFD-snapshots_5_ROM_ANNGPOD)
Diff_CFD_Exp = abs(snapshots_5_Exp-snapshots_5_CFD)
Diff_Exp_ROM = abs(snapshots_5_Exp-snapshots_5_ROM_ANNGPOD)

T_strat_ROM = max(abs(snapshots_5_ROM_ANNGPOD[0,0]-snapshots_5_ROM_ANNGPOD[1,0]),abs(snapshots_5_ROM_ANNGPOD[1,0]-snapshots_5_ROM_ANNGPOD[2,0]),abs(snapshots_5_ROM_ANNGPOD[2,0]-snapshots_5_ROM_ANNGPOD[0,0]))
T_strat_CFD = max(abs(snapshots_5_CFD[0,0]-snapshots_5_CFD[1,0]),abs(snapshots_5_CFD[1,0]-snapshots_5_CFD[2,0]),abs(snapshots_5_CFD[2,0]-snapshots_5_CFD[0,0]))
T_strat_EXP = max(abs(snapshots_5_Exp[0,0]-snapshots_5_Exp[1,0]),abs(snapshots_5_Exp[1,0]-snapshots_5_Exp[2,0]),abs(snapshots_5_Exp[2,0]-snapshots_5_Exp[0,0]))

print("T_strat_ROM is ====", T_strat_ROM)
print("T_strat_CFD is ====", T_strat_CFD)
print("T_strat_EXP is ====", T_strat_EXP)




mdic_error = {"Diff_CFD_ROM" : Diff_CFD_ROM,"Diff_CFD_Exp" : Diff_CFD_Exp, "Diff_Exp_ROM" : Diff_Exp_ROM}             
mdic_Actual = {"snapshots_5_Exp" : snapshots_5_Exp,"snapshots_5_CFD" : snapshots_5_CFD, "snapshots_5_ROM" : snapshots_5_ROM_ANNGPOD}


savemat("ANNGPOD_Error_88_Experiment/ANNGPOD_ExpCFDROM_error_{}_{}_5sensors.mat".format(index,train_size[k]), mdic_error)
savemat("ANNGPOD_Error_88_Experiment/ANNGPOD_ExpCFDROM_actual_{}_{}_5sensors.mat".format(index,train_size[k]), mdic_Actual)





