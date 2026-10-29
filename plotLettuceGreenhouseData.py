# function for loading and plotting lettuce greenhouse benchmark data

# import user-defined help functions
import library as lib 

# import other functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ops       = {}
ops['nx'] = 4    # #of states
ops['nu'] = 3    # #controllable inputs
ops['nd'] = 4    # #disturbance inputs
c         = 86400   

df  = pd.read_csv("data/LettuceGreenhouseBenchmarkData.csv")

nn  = len(df["scenario"].unique())
dfi = df[df['scenario'] == np.random.randint(1,nn+1)]

ops['t'] = dfi['t'].to_numpy()
d        = dfi[['d1','d2','d3','d4']].to_numpy().T
u        = dfi[['u1','u2','u3']].to_numpy().T
x        = dfi[['x1','x2','x3','x4']].to_numpy().T
y        = dfi[['y1','y2','y3','y4']].to_numpy().T



###############################################################################
###############################################################################
###############################################################################
# plot
y_labels = [
    r'$y_1$ (g/m$^2$)',
    r'$y_2$ (ppm) $\cdot 10^3$',
    r'$y_3$ ($^\circ$C)',
    r'$y_4$ (%)'
]

d_labels = [
    r'$d_1$ (W/m$^2$)',
    r'$d_2$ (kg/m$^3$)',
    r'$d_3$ ($^\circ$C)',
    r'$d_4$ (kg/m$^3$)'
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
    a.plot(ops['t'] / c, d[ll, :], drawstyle='steps')
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
