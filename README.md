# RBFPOD & ANNGPOD 88-Sample CFD-ML Framework

This repository contains Python scripts to perform Reduced Order Modeling (ROM) using **RBFPOD** and **ANNGPOD** for CFD data, along with post-processing for error analysis against experimental and CFD results.

---

## 1️⃣ RBFPOD Workflow

### 1.1 Training: `Auto_RBFPOD_88_Train.py`
**Inputs:**
- **Index** – Which of the 88 samples to validate (modify inside the script at lines 36 & 40).  
- **Temp_Tabular** – High-fidelity CFD data:  
  - `T_Highfidelity.npy` (temperature snapshots)  
  - `params0.npy` (parameters)  
- **Percentage_Of_Input_Mod** – Fraction of training points to use.

**Outputs:**
- `Para_test_fixed.npy` – Parameters for validation.  
- `Snapshot_test_fixed.npy` – CFD snapshot for error computation.  
- `ezyrb_RBFPOD_75.rom` – ROM file for predictions.

---

### 1.2 Prediction: `Auto_RBFPOD_88_pred.py`
**Inputs:**
1. Output of `Auto_RBFPOD_88_Train.py`  
2. `File_Vtk` – VTU file to overlay error plots.  
3. `Inputs_to_RBFPOD` – Sensor locations for plotting (26 and 5 sensors).  
4. `Temp_Tabular` – Mesh information:  
   - x, y, z coordinates  
   - Plane mesh sizes (`Size_Mat.mat`)  

**Outputs:**
- MAE on 11 planes → Folder: `RBFPOD_Error_88_Experiment` (`RBF_mae_array_37_75_997.mat`)  
- VTU files of error distribution → `RBFPOD_Error_88_Experiment/RBFPOD_Error_37_75_997`  
- PDF plots comparing CFD vs Exp vs ROM → `RBFPOD_Error_88_Experiment`  
- MAT files for sensor comparison:  
  - `RBFPOD_ExpCFDROM_actual_37_75_26sensors.mat`  
  - `RBFPOD_ExpCFDROM_error_37_75_26sensors.mat`  
  - `RBFPOD_ExpCFDROM_actual_37_75_5sensors.mat`  
  - `RBFPOD_ExpCFDROM_error_37_75_5sensors.mat`  

> **Note:** Do not delete the folder `RBFPOD_Error_88_Experiment`.

---

## 2️⃣ ANNGPOD Workflow

### 2.1 Experiment: `Auto_ANNGPOD_88_Experiment.py`
**Inputs:**
- **Index** – Which of the 88 samples to validate (modify inside the script at lines 34 & 38).  
- **Temp_Tabular** – High-fidelity CFD data (`T_Highfidelity.npy`) and parameters (`params0.npy`).  
- **Percentage_Of_Input_Mod** – Fraction of training points to use.  
- **Experimental Data** – `Exp_Num_Comparison.mat` (5 sensor measurements).  
- **File_Vtk** – VTU file to overlay error plots.  
- **Inputs_to_ANNGPOD** – Sensor information for plotting (26 and 5 sensors).  
- **Temp_Tabular** – Mesh info:  
  - x, y, z coordinates  
  - Plane sizes (`Size_Mat.mat`)  

**Outputs:**
- MAE on 11 planes → Folder: `ANNGPOD_Error_88_Experiment`  
- VTU files of error distribution → `ANNGPOD_Error_88_Experiment/ANNGPPOD_Error_37_75_997`  
- PDF plots comparing CFD vs Exp vs ROM → `ANNGPOD_Error_88_Experiment`  
- MAT files for sensor comparison:  
  - `ANNGPOD_ExpCFDROM_actual_37_75_26sensors.mat`  
  - `ANNGPOD_ExpCFDROM_error_37_75_26sensors.mat`  
  - `ANNGPOD_ExpCFDROM_actual_37_75_5sensors.mat`  
  - `ANNGPOD_ExpCFDROM_error_37_75_5sensors.mat`  

---

## 3️⃣ How to Run

1. **RBFPOD Training:**  
```bash
python Auto_RBFPOD_88_Train.py

2. **RBFPOD Prediction:**  
```bash
python Auto_RBFPOD_88_pred.py

3. **ANNGPOD Training & Prediction::**  
```bash
python Auto_ANNGPOD_88_Experiment.py

