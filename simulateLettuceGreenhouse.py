# import user-defined help functions
import library as lib 

# import other functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df    = []
N0    = np.array([0, 20448, 40896, 61334, 81504])          # initial samples different scenarios
#N0    = np.array([0])                                      # pick one scenario scenarios
casus = ['train','validation','train','test','train']      # case corresponding to element in N0
for nn in range(len(N0)):
    
    dfi   = pd.DataFrame(columns=['scenario', 'casus', 't', 'd1', 'd2', 'd3', 'd4', 'u1', 'u2', 'u3', 'x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4'])
    
    # model parameters   
    p         = lib.modelParameter()
    
    ops       = {}
    ops['nx'] = 4    # #of states
    ops['nu'] = 3    # #controllable inputs
    ops['nd'] = 4    # #disturbance inputs
    
    # simulation parameters
    c         = 86400
    nDays     = 40                                      # #days in simulation (max = 46 weeks)
    ops['h']  = 15*60                                   # sample period in seconds: #minutes*60
    ops['L']  = nDays*c                                 # final time simulation
    ops['t']  = np.arange(0,ops['L']+ops['h'],ops['h']) # initial time vector
    ops['N']  = len(ops['t'])                           # number of samples in one simulation    
    ops['N0'] = N0[nn]                                  # initial sample number disturbance
    
    # inititialize mpc
    ops       = lib.initMpc(p,ops)
    
    # signals
    d         = lib.disturbance(ops)
    x         = np.zeros((ops['nx'],ops['N']+1))
    x[:,0]    = np.array([0.0035, 1e-03, 15, 0.008])
    y         = np.zeros((ops['nx'],ops['N']))
    u         = np.zeros((ops['nu'],ops['N']+1))
    
    for kk in range(ops['N']):
        
        # simulation
        x[:,kk+1] = lib.fd(x[:,kk], u[:,kk], d[:,kk], p, ops['h'])
        y[:,kk]   = lib.g(x[:,kk], u[:,kk], d[:,kk], p, ops['h'])
        
        # control
        ops       = lib.updateConstraint(x[:,kk], u[:,kk], d[:,kk], p, ops)
        u[:,kk+1] = lib.controller(x[:,kk], d[:,kk:kk+ops['Np']], p, ops)
       

    dfi['t']                   = ops['t']
    dfi[['d1','d2','d3','d4']] = d[:,0:ops['N']].T
    dfi[['u1','u2','u3']]      = u[:,0:ops['N']].T
    dfi[['x1','x2','x3','x4']] = x[:,0:ops['N']].T
    dfi[['y1','y2','y3','y4']] = y[:,0:ops['N']].T
    
    dfi['scenario']            = nn+1 
    dfi['casus']               = casus[nn]
    
    df.append(dfi)
    
df = pd.concat(df,ignore_index=True)

df.to_csv("data/lettuceGreenhouseBenchmarkData.csv", index=False)





###############################################################################
###############################################################################
###############################################################################

dfi = df[df['scenario'] == np.random.randint(1,nn+1)]

d   = dfi[['d1','d2','d3','d4']].to_numpy().T
u   = dfi[['u1','u2','u3']].to_numpy().T
x   = dfi[['x1','x2','x3','x4']].to_numpy().T
y   = dfi[['y1','y2','y3','y4']].to_numpy().T

# convert outside Co2 and humidity to desired units for plotting
dc      = np.zeros(d.shape)
dc[0,:] = d[0,:] 
dc[1,:] = lib.co2dens2ppm(d[2,:], d[1,:])*1e-3
dc[2,:] = d[2,:]
dc[3,:] = lib.vaporDens2rh(d[2,:], d[3,:])

# plot
y_labels = [
    r'$y_1$ (g/m$^2$)',
    r'$y_2$ (ppm) $\cdot 10^3$',
    r'$y_3$ ($^\circ$C)',
    r'$y_4$ (\%)'
]

d_labels = [
    r'$d_1$ (W/m$^2$)',
    r'$d_2$ (ppm) $\cdot 10^3$',
    r'$d_3$ ($^\circ$C)',
    r'$d_4$ (\%)'
]

u_labels = [
    r'$u_1$ (mg/m$^2$/s)',
    r'$u_2$ (mm/s) $\cdot 10^3$',
    r'$u_3$ (W/m$^2$)'
]

n_left = ops['nd'] + ops['nu']
nrows = max(n_left, ops['nx'])

fig, ax = plt.subplots(nrows, 2, figsize=(16, 2 * nrows), sharex=True)

fig.suptitle(
    f"{dfi.casus.unique().item()}: scenario {dfi.scenario.unique().item()}",
    fontsize=23,
    y=0.99
)

# Left column: disturbances d, followed by inputs u
for ll in range(ops['nd']):
    a = ax[ll, 0]
    a.plot(ops['t'] / c, dc[ll, :], drawstyle='steps')
    a.set_ylabel(d_labels[ll], fontsize=18)
    a.grid(True)

for ll in range(ops['nu']):
    row = ops['nd'] + ll
    a = ax[row, 0]
    a.plot(ops['t'] / c, u[ll, :], drawstyle='steps')
    a.set_ylabel(u_labels[ll], fontsize=18)
    a.grid(True)

# Right column: outputs y
for ll in range(ops['nx']):
    a = ax[ll, 1]
    a.plot(ops['t'] / c, y[ll, :], drawstyle='steps')
    a.set_ylabel(y_labels[ll], fontsize=18)
    a.grid(True)

    if ll == 3:
        a.set_ylim(0, 100)
    else:
        a.set_ylim(bottom=0, top=1.1 * max(y[ll, :]))

# Hide unused panels
for ll in range(nrows):
    if ll >= n_left:
        ax[ll, 0].set_visible(False)
    if ll >= ops['nx']:
        ax[ll, 1].set_visible(False)

# X-axis and labels
for col in range(2):
    visible_axes = [ax[ll, col] for ll in range(nrows)
                    if ax[ll, col].get_visible()]

    for a in visible_axes:
        a.set_xlim(ops['t'][0] / c, ops['t'][-1] / c)

    if visible_axes:
        visible_axes[-1].set_xlabel('Time (days)', fontsize=18)

# Reduce space above the first row
fig.tight_layout(rect=[0, 0, 1, 0.985], pad=0.5, h_pad=0.5, w_pad=2)
    

