print("Let's do this thing", flush = True)

## Load in packages

try:
    import pandas as pd
    import numpy as np
    from scipy.special import kv, gamma
    import math
    
    import argparse
    from copy import deepcopy
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import KFold
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.metrics import mean_squared_error
    from torch.utils.data import DataLoader, TensorDataset

except:
    print("loading packages failed", flush = True)

import pandas as pd
import numpy as np
from scipy.special import kv, gamma
import math

import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset



# We need to sort out the arguments passed from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--spatial_model', type=str, required=True)
parser.add_argument('--temporal_model', type=str, required=True)
parser.add_argument('--network_architecture', type=int, required=True)
parser.add_argument('--p_dropout', type=float, required=True)
args = parser.parse_args()

spatial_model, temporal_model, network_architecture, p_dropout = args.spatial_model, args.temporal_model, args.network_architecture, args.p_dropout
print(f"spatial_model = {spatial_model}, temporal_model = {temporal_model}, network architecture = {network_architecture}, dropout probability = {p_dropout}", flush = True)

# Matern function
def matern(d, nu, l=1):
    factor = (2**(1-nu)) / gamma(nu)
    scaled_d = np.sqrt(2 * nu) * d / l
    if np.isnan(factor * (scaled_d**nu) * kv(nu, scaled_d)):  # Check for NaN
        print("NaN detected, d:", d)
        return 1
    return factor * (scaled_d**nu) * kv(nu, scaled_d)

    
checkpoint = 0

## Load in data
df = pd.read_csv('data/50_perc_monthly_rain_1981-2010.csv')
df = df.sort_values(by=["t", "stno"])

# Adjust time index
df["t"] = df["t"] - 1



## Setup the data
df = df.reset_index(drop=True)
nt = df["t"].max() + 1
ns = df["stno"].nunique()

# Get spatial coordinates and normalize
lon = df[df["t"] == 0]["east"].values
lat = df[df["t"] == 0]["north"].values
normalized_lon = (lon-min(lon))/(max(lon)-min(lon))
normalized_lat = (lat-min(lat))/(max(lat)-min(lat))
N = df.shape[0]

# Time coordinates (you can normalize them)
t = np.arange(nt)



## Space basis functions (now using mesh)

checkpoint = 1
print(f'Checkpoint {checkpoint}', flush = True)

# Meshes at different resolutions
mesh1 = pd.read_csv('data/meshes/rain_50_station_mesh.csv', index_col = 0)
mesh2 = pd.read_csv('data/meshes/rain_50_station_mesh_res2.csv', index_col = 0)
mesh3 = pd.read_csv('data/meshes/rain_50_station_mesh_res3.csv', index_col = 0)
mesh4 = pd.read_csv('data/meshes/rain_50_station_mesh_res4.csv', index_col = 0)
meshes = [mesh1, mesh2, mesh3, mesh4]

# Normalize the meshes
for idx, mesh in enumerate(meshes):
    
    meshes[idx] = meshes[idx].to_numpy()
    meshes[idx][:, 0] = (meshes[idx][:, 0] - min(lon))/(max(lon)-min(lon))
    meshes[idx][:, 1] = (meshes[idx][:, 1] - min(lat))/(max(lat)-min(lat))

# Memory for spatial basis functions
phi_s = np.zeros((len(lon), sum([len(mesh) for mesh in meshes])))
basis_size = 0

# Wendland spatial
if spatial_model == "Wendland":
    theta = 2.5
    avg_dists = [12, 30, 70, 150]
    for res in range(len(meshes)):
        theta_res = 1/avg_dists[res] * theta
        knots_x, knots_y = meshes[res][:, 0], meshes[res][:, 1]
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        
        for i in range(len(meshes[res])):
            d = np.linalg.norm(np.vstack((normalized_lon,normalized_lat)).T-knots[i,:],axis=1)/theta_res
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi_s[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi_s[j,i + basis_size] = 0
        basis_size = basis_size + len(meshes[res])
        print(f'Spatial resolution {res} done', flush = True)
    phi_s[phi_s < 1e-6] = 0

# Gaussian Radial Basis functions
elif spatial_model == "GBRF":
    theta = 2.5
    avg_dists = [12, 30, 70, 150]
    length_scales = [0.5, 1, 2, 10]
    for res in range(len(meshes)):
        theta_res, length_scale = 1/avg_dists[res] * theta, length_scales[res]
        knots_x, knots_y = meshes[res][:, 0], meshes[res][:, 1]
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        for i in range(len(meshes[res])):
            d = np.linalg.norm(np.vstack((normalized_lon,normalized_lat)).T-knots[i,:],axis=1) / theta_res
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi_s[j,i + basis_size] = np.exp(-length_scale * d[j])
                else:
                    phi_s[j,i + basis_size] = 0
        basis_size = basis_size + len(meshes[res])
        print(f'Spatial resolution {res} done', flush = True)
    phi_s[phi_s < 1e-6] = 0

# Matern
elif spatial_model == "Matern":
    theta = 2.5
    s_l = 0.5
    avg_dists = [12, 30, 70, 150]
    s_nus = [2.5, 2.5, 2.5, 2.5]
    
    for res in range(len(meshes)):
        theta_res = 1/avg_dists[res] * theta
        knots_x, knots_y = meshes[res][:, 0], meshes[res][:, 1]
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        s_nu = s_nus[res]
        
        for i in range(len(meshes[res])):
            d = np.linalg.norm(np.vstack((normalized_lon,normalized_lat)).T-knots[i,:],axis=1)/theta_res
            for j in range(len(d)):
                phi_s[j,i + basis_size] = matern(d[j], s_nu, s_l)
        basis_size = basis_size + len(meshes[res])
        print(f'Spatial resolution {res} done', flush = True)
    phi_s[phi_s < 1e-6] = 0
    
# Not doing kronecker, so need to duplicate basis functions for each time
phi_s = np.tile(phi_s, (nt, 1))
checkpoint += 1
print(f'Checkpoint {checkpoint}', flush = True)

## Time basis functions

# Make knots and assign memory
num_basis = [20, 77, 331]
temporal_knots = [np.linspace(0,nt-1,int(i)) for i in num_basis]
phi_t = np.zeros((nt, sum(num_basis)))
K = 0

# Wendland temporal
if temporal_model == "Wendland":
    std_arr = np.array([8, 5, 1])
    for res in range(len(num_basis)):
        std = std_arr[res]
        for i in range(num_basis[res]):
            d = np.absolute(t-temporal_knots[res][i]) / (std**2)
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi_t[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi_t[j,i + K] = 0
        K = K + num_basis[res]
        print(f'temporal resolution {res} done', flush = True)
    phi_t[phi_t < 1e-6] = 0

# Gaussian Radial Basis functions
elif temporal_model == "GBRF":
    std_arr = np.array([8, 5, 1])
    length_scales = [0.3, 0.7, 2]
    for res in range(len(num_basis)):
        std, length_scale = std_arr[res], length_scales[res]
        for i in range(num_basis[res]):
            d = np.absolute(t-temporal_knots[res][i]) / (std**2)
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi_t[j,i + K] = np.exp(-length_scale * d[j])
                else:
                    phi_t[j,i + K] = 0
        K = K + num_basis[res]
        print(f'Temporal resolution {res} done', flush = True)
    phi_t[phi_t < 1e-6] = 0

# Matern
elif temporal_model == "Matern":
    t_nus = [4, 4, 4]
    t_l = 0.5
    
    std_arr = np.array([8, 5, 1])
    for res in range(len(num_basis)):
        std = std_arr[res]
        t_nu = t_nus[res]
        for i in range(num_basis[res]):
            d = np.absolute(t-temporal_knots[res][i]) / (std**2)
            for j in range(len(d)):
                phi_t[j,i + K] = matern(d[j], t_nu, t_l)
        K = K + num_basis[res]
        print(f'temporal resolution {res} done', flush = True)
    phi_t[phi_t < 1e-6] = 0




# This is for non-Kronecker case (duplicate and append)
phi_t = np.repeat(phi_t, ns, axis=0)
phi = np.hstack((phi_s, phi_t))
checkpoint += 1
print(f'Checkpoint {checkpoint}', flush = True)

# Remove basis functions for missing data
missing_idx = df[df["qc_rain"].isna()].index
phi = np.delete(phi, missing_idx, axis=0)
checkpoint += 1
print(f'Checkpoint {checkpoint}', flush = True)

# Remove all-zero columns
idx_zero = np.where(np.all(phi == 0, axis=0))[0]
phi_reduce = np.delete(phi,idx_zero,1)
print(phi.shape, flush = True)
print(phi_reduce.shape, flush = True)
checkpoint += 1
print(f'Checkpoint {checkpoint}', flush = True)

print("Basis functions done.", flush = True)




## Set up inputs for neural network
phi_obs = phi_reduce

# Matrix of covariates X
x_idx = [df.columns.get_loc('east'), df.columns.get_loc('north'), \
         df.columns.get_loc('points5'), df.columns.get_loc('exp25k'), \
         df.columns.get_loc('e5'), df.columns.get_loc('n5'), \
         df.columns.get_loc('s5'), df.columns.get_loc('w5'), \
         df.columns.get_loc('dist2c'), df.columns.get_loc('nw5'), \
         df.columns.get_loc('ne5'), df.columns.get_loc('se5'), \
         df.columns.get_loc('sw5'), df.columns.get_loc('elev'), \
         df.columns.get_loc('t')]
X = df.iloc[~df.index.isin(missing_idx), x_idx].values

# Rainfall values (Transformed with square root)
y_idx = df.columns.get_loc('qc_rain')
y = df.iloc[~df.index.isin(missing_idx), y_idx].values
y_sqrt = np.sqrt(y)

# Normalize covariates
normalized_X = X
for i in range(X.shape[1]):
    normalized_X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))
N_obs = X.shape[0]

inputs = np.hstack((normalized_X, phi_obs))
targets = y




# Metric functions
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)




# Network architectures
def make_model_1(p, p_dropout):
    model = nn.Sequential(
        nn.Linear(p, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p_dropout),

        nn.Linear(128, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p_dropout),

        nn.Linear(128, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p_dropout),

        nn.Linear(128, 128), 
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p_dropout),

        nn.Linear(128, 64), 
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p_dropout),

        nn.Linear(64, 1)
    )
    return model

def make_model_2(p, p_dropout):
    model = nn.Sequential(
        nn.Linear(p, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 512), 
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p_dropout),

        nn.Linear(512, 1024), 
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p_dropout),

        nn.Linear(1024, 2048), 
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Dropout(p_dropout),

        nn.Linear(2048, 1024), 
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p_dropout),

        nn.Linear(1024, 512), 
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p_dropout),

        nn.Linear(512, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p_dropout),

        nn.Linear(64, 1)
    )
    return model


def make_model_3(p, p_dropout):
    model = nn.Sequential(
        nn.Linear(p, 512), 
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p_dropout),

        nn.Linear(512, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 256), 
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p_dropout),

        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(p_dropout),

        nn.Linear(64, 1)
    )
    return model


def make_model(network_architecture, p, p_dropout):
    if network_architecture == 1:
        return make_model_1(p, p_dropout)
    elif network_architecture == 2:
        return make_model_2(p, p_dropout)
    elif network_architecture == 3:
        return make_model_3(p, p_dropout)
    else:
        print("Provide a valid architecture", flush = True)
        return None
        



## Hyperparameters and setup
epochs = 400
learning_rate = 1e-3
batch_size = 512

seed = 222
torch.manual_seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state = seed)

# Convert numpy arrays to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

# Vector of scores
train_rmse_scores = []
val_rmse_scores = []

# Array for comparing true values to predicted ones
y_obs_and_pred = np.zeros((len(targets_tensor), 2))
y_obs_and_pred[:, 0] = targets_tensor.numpy().flatten()



## Training
fold = 0
for train_index, val_index in kf.split(inputs_tensor):
    
    fold += 1
    print(f"Fold {fold}", flush = True)

    best_rmse = float('inf')
    
    # Split data into train and validation sets
    train_inputs, val_inputs = inputs_tensor[train_index], inputs_tensor[val_index]
    train_targets, val_targets = targets_tensor[train_index], targets_tensor[val_index]
    
    # Create DataLoader for batching
    train_data = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = make_model(network_architecture = network_architecture, p=train_inputs.shape[1], p_dropout=p_dropout)
    criterion = nn.MSELoss()
    weight_decay = 1e-3  # L2 regularization strength
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}", flush = True)
        model.train()
        running_loss = 0.0
    
        for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
        # Compute RMSE for training data
        model.eval()
        with torch.no_grad():
            train_outputs = model(train_inputs)
            train_mse = mean_squared_error(train_targets.numpy(), train_outputs.numpy())
            train_rmse = np.sqrt(train_mse)
    
            val_outputs = model(val_inputs)
            val_mse = mean_squared_error(val_targets.numpy(), val_outputs.numpy())
            val_rmse = np.sqrt(val_mse)
    
        # Print RMSE for training and validation data
        print(f"Average Loss: {running_loss / len(train_loader):.4f}", flush = True)
        print(f"Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}", flush = True)
    
        if train_rmse < best_rmse:
            best_rmse = train_rmse
            checkpoint = deepcopy(model.state_dict())
            print(f'best_rmse = {best_rmse}')
    
    # Validation loop
    model.load_state_dict(checkpoint)
    model.eval()

    
    with torch.no_grad():
        val_outputs = []
        val_targets_list = []
        for val_inputs_batch, val_targets_batch in val_loader:
            outputs = model(val_inputs_batch)
            val_outputs.append(outputs)
            val_targets_list.append(val_targets_batch)
        
        val_outputs = torch.cat(val_outputs, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)
        
        mse = mean_squared_error(val_targets.numpy(), val_outputs.numpy())
        rmse = np.sqrt(mse)

    y_obs_and_pred[val_index, 1] = val_outputs.numpy().squeeze()
    
    print(f"Fold {fold} - MSE: {mse}, RMSE: {rmse}", flush = True)



## Final results
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
    
print(f"RMSE: {rmse(y_obs_and_pred[:, 0], y_obs_and_pred[:, 1])}", flush = True)
print(f"R_Squared: {r_squared(y_obs_and_pred[:, 0], y_obs_and_pred[:, 1])}", flush = True)

df = df.dropna(subset=["qc_rain"])
df["ypred"] = y_obs_and_pred[:, 1]

print(f"spatial_model = {spatial_model}, temporal_model = {temporal_model}, network architecture = {network_architecture}, dropout probability = {p_dropout}", flush = True)

file_destination = f"../scratch/results/rain_50_ST_DK_s{spatial_model}_t{temporal_model}_n{network_architecture}.csv"
df.to_csv(file_destination, index=False)

print("CV Finished")