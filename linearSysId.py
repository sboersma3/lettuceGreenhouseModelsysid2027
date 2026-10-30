# import user-defined help functions
import library as lib 

# import other functions
import numpy as np
import pandas as pd
import pysindy as ps

import pdb #pdb.set_trace()

ops       = {}
ops['nx'] = 4    # #of states
ops['nu'] = 3    # #controllable inputs
ops['nd'] = 4    # #disturbance inputs
c         = 86400 
    
###############################################################################
## lettuce greenhouse linear system identification using 
## Sparse Identification of Nonlinear Dynamics (SINDy)
###############################################################################

### define linear library
library   = {}
variables = ["x1", "x2", "x3","w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11"]

for i, var in enumerate(variables, start=1):
    library[f'library_functions_{i}'] = [lambda x: x]
    library[f'library_function_names_{i}'] = [lambda x, i=i: f"f{i}({x})"]
    
Nlib           = 11
inputs_per_lib = [[i]*Nlib for i in range(Nlib)]

custom_library_ll  = [None] * Nlib
for ll in range(Nlib):
       
    custom_library_ll[ll] = ps.CustomLibrary(
        library_functions=library['library_functions_'+str(ll+1)], function_names=library['library_function_names_'+str(ll+1)],
        interaction_only=True, include_bias=False)
    
custom_library = ps.GeneralizedLibrary(custom_library_ll, inputs_per_library=inputs_per_lib)
feature_names  = ["x1(t)","x2(t)","x3(t)","x4(t)"]       
feature_names  = feature_names + ["u1(t)","u2(t)","u3(t)","d1(t)","d2(t)","d3(t)","d4(t)"]

### load data
DF  = pd.read_csv('data/lettuceGreenhouseBenchmarkData.csv')

# select one period of the data
N0  = 10                        # start day
N1  = min(20,DF["t"].max()/c)   # end day
df  = DF[(DF["t"] >= N0 * 86400) & (DF["t"] <= N1 * 86400)].copy()


df_train = df[df['casus'] == 'train']
df_val   = df[df['casus'] == 'validation']
df_test  = df[df['casus'] == 'test']

n_train  = df_train['scenario'].unique()
h        = df['t'].diff().median()

T,D,U,X,Y,W  = [],[],[],[],[],[]
for nn in n_train:           
   
    dfi = df_train[df_train['scenario'] == nn]

    t   = dfi['t'].to_numpy()
    d   = dfi[['d1','d2','d3','d4']].to_numpy().T
    u   = dfi[['u1','u2','u3']].to_numpy().T
    x   = dfi[['x1','x2','x3','x4']].to_numpy().T
    y   = dfi[['y1','y2','y3','y4']].to_numpy().T

    T   = T + [t]
    D   = D + [d.T]     
    U   = U + [u.T]
    X   = X + [x.T]
    Y   = Y + [y.T]  
    W   = W + [np.column_stack([u.T,d.T])]
    

### optimize SINDy model
opt   = ps.STLSQ(threshold=0*1e-32, normalize_columns=False)
model = ps.SINDy(feature_library=custom_library, optimizer=opt)

model.fit(Y,u=W,t=h,feature_names=feature_names)

# plot training data
lib.plotTrainingData(D,U,X,Y,T,ops)

# ###############################################################################
# ###############################################################################
            
# # test the model on training data
# qq       = np.random.randint(ops['Ns'])
# x        = X[qq]
# y        = Y[qq]
# u        = U[qq]
# d        = D[qq]
# w        = W[qq]
# t        = T[qq]
# h        = np.diff(t)

# xhat           = np.zeros((ops['N']+1,ops['nx']))
# xhat[0:Np+1,:] = y[0:Np+1,:]

# for kk in range(Np,ops['N']-1): # simulate the model numerically
        
#     xhat[kk+1,:]  = xhat[kk,:] + h[kk]*model.predict(xhat[[kk],:], w[kk,:])
    
    
#     if Np>0:
#         w[kk+1,0:ops['nx']] = xhat[kk+1,:]

# yhat = xhat
 
# # plot validation on training data
# lib.plotTrainingValidation(y,yhat,u,d,ops)

# ###############################################################################
# ###############################################################################

# # test the model on test data

# w           = np.column_stack((u.T,d.T))
# x           = sol.y.T 
# y           = np.vstack([x[:,0]*1e3, lib.co2dens2ppm(x[:,2],x[:,1]), x[:,2], lib.vaporDens2rh(x[:,2],x[:,3])]).T   
# u           = u.T
# d           = d.T
# t           = sol.t
# h           = np.diff(t)

# xhat           = np.zeros((ops['N']+1,ops['nx']))
# xhat[0:Np+1,:] = y[0:Np+1,:]

# for kk in range(Np,ops['N']-1): # simulate the model numerically
        
#     xhat[kk+1,:]  = xhat[kk,:] + h[kk]*model.predict(xhat[[kk],:], w[kk,:])
    
    
#     if Np>0:
#         w[kk+1,0:ops['nx']] = xhat[kk+1,:]


# yhat = xhat

# # plot validation on test data
# lib.plotTestValidation(y,yhat,u,d,ops)






# ###############################################################################
# ###############################################################################

# phat = model.coefficients()
# A    = phat[:,0:4]
# B    = phat[:,4:7]
# Bd   = phat[:,7:]
# C    = np.eye(ops['nx'])
# D    = np.zeros((ops['nx'],ops['nu']))

# xhat_           = np.zeros((ops['N']+1,ops['nx']))
# xhat_[0:Np+1,:] = y[0:Np+1,:]

# for kk in range(Np,ops['N']-1): # simulate the model numerically
        
#     xhat_[kk+1,:]  = xhat_[kk,:] + h[kk]*( A.dot(xhat_[kk,:].T) + B.dot(u[kk,:].T) + Bd.dot(d[kk,:]) )
    

# yhat_ = xhat_


# fig, ax = plt.subplots(ops['nx'])
# fig.set_size_inches(10.5, 10.5)
# for ll in range(ops['nx']):
#     ax[ll].plot(ops['t']/c,yhat[0:-1,ll], drawstyle='steps')
#     ax[ll].plot(ops['t']/c,yhat_[0:-1,ll], drawstyle='steps', linestyle='--', color='red')
#     ax[ll].grid(True)
#     ax[ll].set_xlim(left=ops['t'][0], right=ops['t'][-1]/c)
#     if ll==0:
#         ax[ll].set_ylabel(r'$y_1$ (g/m$^2$)', fontsize=16)
#         ax[ll].set_title(r'linear vs sindy model (should be equal)', fontsize=16)
#     elif ll==1:
#         ax[ll].set_ylabel(r'$y_2$ (ppm)', fontsize=16)
#     elif ll==2:   
#         ax[ll].set_ylabel(r'$y_3$ ($^\circ$C)', fontsize=16)
#     elif ll==3:
#         ax[ll].set_ylabel(r'$y_4$ (%)', fontsize=16)
#         ax[ll].set_xlabel(r'Time (days)',fontsize=16) 


# ###############################################################################
# ###############################################################################
