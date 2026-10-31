import numpy as np
import matplotlib.pyplot as plt
import math
import casadi 
import scipy.io as sio
from scipy import signal
import statistics  

rng = np.random.default_rng(23)

import pdb #pdb.set_trace()  

 
def modelParameter():
# Model parameters
    # parameter 					description 									[unit] 					nominal value
    p = {'satH2O1': 9348} 		    # saturation water vapour parameter 			[J m^{-3}] 				9348
    p['satH2O2'] = 17.4 			# saturation water vapour parameter 			[-] 					17.4
    p['satH2O3'] = 239 		        # saturation water vapour parameter 			[°C] 					239
    p['satH2O4'] = 10998  	        # saturation water vapour parameter 			[J m^{-3}] 				10998
    p['R'] = 8314 					# ideal gas constant 							[J K^{-1} kmol^{-1}] 	8314
    p['T'] = 273.15 				# conversion from C to K 						[K] 					273.15

    p['leak'] = 0.75e-4 			# ventilation leakage through the cover 		[m s^{-1}] 				0.75e-4
    p['CO2cap'] = 4.1 				# CO2 capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1
    p['H2Ocap'] = 4.1 				# Vapor capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1
    p['aCap'] = 3e4 				# effective heat capacity of the greenhouse air [J m^{-2}{gh} °C^{-1}]  3e4
    p['ventCap'] = 1290 			# heat capacity per volume of greenhouse air 	[J m^{-3}{gh} °C^{-1}]  1290
    p['trans_g_o'] = 6.1 			# overall heat transfer through the cover 		[W m^{-2}{gh} °C^{-1}]  6.1
    p['rad_o_g'] = 0.2 			    # heat load coefficient due to solar radiation 	[-] 					0.2
    
    p['alfaBeta'] = 0.544 		    # yield factor 									[-] 					0.544
    p['Wc_a'] = 2.65e-7			    # respiration rate 								[s^{-1}] 				2.65e-7
    p['CO2c_a'] = 4.87e-7 		    # respiration coefficient 						[s^{-1}]  				4.87e-7
    p['laiW'] = 53 				    # effective canopy surface 						[m^2{leaf} kg^{-1}{dw}] 53
    p['photI0'] = 3.55e-9 	        # light use efficiency 							[kg{CO2} J^{-1}]  		3.55e-9
    p['photCO2_1'] =5.11e-6  	    # temperature influence on photosynthesis 		[m s^{-1} °C^{-2}] 		5.11e-6
    p['photCO2_2'] =2.3e-4		    # temperature influence on photosynthesis 		[m s^{-1} °C^{-1}] 		2.3e-4
    p['photCO2_3'] =6.29e-4 		# temperature influence on photosynthesis 		[m s^{-1}] 				6.29e-4
    p['photGamma'] = 5.2e-5 		# carbon dioxide compensation point 			[kg{CO2} m^{-3}{air}] 	5.2e-5
    p['evap_c_a'] = 3.6e-3 		    # coefficient of leaf-air vapor flow 			[m s^{-1}] 				3.6e-3
       
    return p


def disturbance(ops):

    c       = 86400
    nDays   = ops['L']/c                        # number of days in the simulation
    D       = sio.loadmat('data/outdoorWeatherWurGlas2014.mat')
    D       = D['d']
    
    t       = D[:,0]                            # Time [days]
    t       = t - t[0]  
    
    dt      = statistics.mean(np.diff(t))       # Sample period data [days]
    Ns      = math.ceil(nDays/dt)               # Number of samples we need
    N0      = ops['N0']                         # Start sample
    if Ns>len(t):
       print(' ') 
       print('Not enough samples in the data.')
       print(' ')
    
    # extract only data for current simulation
    t       = D[N0:N0+Ns-1,0]*c                 # Time [s]
    t       = t - t[0] 
    dt      = statistics.mean(np.diff(t))       # Sample period data [s]
    
    if ops['h']<dt:
       print(' ') 
       print('Increase ops[h], sample period too small.')
       print(' ')
    
    # new sample period p times the original sample rate
    p       = math.floor(1/(dt/ops['h']))
    
    rad     = D[N0:N0+Ns+p*ops['Np'],1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns+p*ops['Np'],2]          # Outdoor temperature [°C]
    co2Out  = D[N0:N0+Ns+p*ops['Np'],5]          # Outdoor CO2 concentration [ppm]
    co2Out  = co2ppm2dens(tempOut,co2Out)        # Outdoor CO2 concentration [kg/m3]
    vapOut  = D[N0:N0+Ns+p*ops['Np'],3]          # Outdoor relative humidity [#]
    vapOut  = rh2vaporDens(tempOut,vapOut)       # Outdoor humidity [kg/m3]
        
    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])
    
    ns              = math.ceil(len(rad)/p)
    d               = np.zeros((ops['nd'],ns))
            
    d[0,:]          = signal.resample(rad-d0[0],ns)+d0[0]  
    d[1,:]          = signal.resample(co2Out-d0[1],ns)+d0[1]          
    d[2,:]          = signal.resample(tempOut-d0[2],ns)+d0[2] 
    d[3,:]          = signal.resample(vapOut-d0[3],ns)+d0[3]      
    
    d[0,d[0,:]<0]   = 0
    
    #plt.plot(ops['t']/c,d[0,0:ns-ops['Np']-1])
    #plt.plot(t/c,rad[0:Ns-1],'--')
    #plt.savefig("test.svg")
    #plt.show()
    
    return d

def fd(x,u,d,p,h):
    
    # x(1)    = x.dw
    # x(2)    = x.co2In
    # x(3)    = x.tempIn
    # x(4)    = x.vapIn
    
    # d(1)    = d.rad
    # d(2)    = d.co2Out
    # d(3)    = d.tempOut
    # d(4)    = d.vapOut
    
    # u(1)    = u.co2
    # u(2)    = u.vent
    # u(3)    = u.heat
    
    k1  = f(x,u,d,p,h)
    k2  = f(x + h/2 * k1,u,d,p,h)
    k3  = f(x + h/2 * k2,u,d,p,h)
    k4  = f(x + h * k3,u,d,p,h)
    x   = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    dx  = x * ( 1 + .007*rng.uniform(-1, 1, x.shape[0]) )
    
    return dx

def f(x,u,d,p,h):
        
    ki =  np.array([p['alfaBeta']*( \
        (1-np.exp(-p['laiW']*x[0]))*p['photI0']*d[0]* \
        (                  -p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']) \
        /(p['photI0']*d[0]+(-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']))) \
        - p['Wc_a']*x[0]*2**(0.1*x[2]-2.5), \
        1/p['CO2cap']*(-((1-np.exp(-p['laiW']*x[0]))*p['photI0']*d[0]* \
        (-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']) \
        /(p['photI0']*d[0]+(-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']))) \
        + p['CO2c_a']*x[0]*2**(0.1*x[2]-2.5) + u[0]/1e6 - (u[1]/1e3+p['leak'])*(x[1]-d[1])),        
        1/p['aCap']*(u[2] - (p['ventCap']*u[1]/1e3+p['trans_g_o'])*(x[2]-d[2]) + p['rad_o_g']*d[0]),        
        1/p['H2Ocap']*((1-np.exp(-p['laiW']*x[0]))*p['evap_c_a']*(p['satH2O1']/(p['R']*(x[2]+p['T']))* \
        np.exp(p['satH2O2']*x[2]/(x[2]+p['satH2O3']))-x[3]) - (u[1]/1e3+p['leak'])*(x[3]-d[3]))])    

    return ki
    
def g(x,u,d,p,h):
    
    y = np.array([1e3*x[0],1e-3*co2dens2ppm(x[2],x[1]), x[2], vaporDens2rh(x[2], x[3])]) * ( 1 + .0053*rng.uniform(-1, 1, x.shape[0]) )
         
    return y

def rh2vaporDens(temp,rh):
   
    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])
    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1]))) 
    # Saturation vapor pressure of air in given temperature [Pa]   
    pascals=(rh/100)*satP # Partial pressure of vapor in air [Pa]
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    vaporDens = np.divide(pascals*Mw,(R*(temp+C2K)))
        
    return vaporDens
   
def vaporDens2rh(temp,vaporDens):
      
    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])  
    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1]))) 
    # Saturation vapor pressure of air in given temperature [Pa]      
    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.    
    if isinstance(temp, float) or isinstance(temp, int) : 
        rh = min(100.0,np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens)
    else:
        rh = np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens
    
    return rh

def co2dens2ppm(temp,dens):
        
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]   
    ppm = 1e6*R*(temp+C2K)*dens/(P*M_CO2)
    
    return ppm

def co2ppm2dens(temp, ppm):
            
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]   
    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    co2Dens = np.divide(P*1e-6*ppm*M_CO2,(R*(temp+C2K)))
    
    return co2Dens

###############################################################################
###############################################################################

               ####### controller functions ############

###############################################################################
###############################################################################

def fdhat(x,u,d,p,h):
    
    # x(1)    = x.dw
    # x(2)    = x.co2In
    # x(3)    = x.tempIn
    # x(4)    = x.vapIn
    
    # d(1)    = d.rad
    # d(2)    = d.co2Out
    # d(3)    = d.tempOut
    # d(4)    = d.vapOut
    
    # u(1)    = u.co2
    # u(2)    = u.vent
    # u(3)    = u.heat
    
    k1  = f(x,u,d,p,h)
    k2  = f(x + h/2 * k1,u,d,p,h)
    k3  = f(x + h/2 * k2,u,d,p,h)
    k4  = f(x + h * k3,u,d,p,h)
    x   = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    dx  = x
    
    return dx

def fhat(x,u,d,p,h):
        
    ki =  np.array([p['alfaBeta']*( \
        (1-np.exp(-p['laiW']*x[0]))*p['photI0']*d[0]* \
        (                  -p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']) \
        /(p['photI0']*d[0]+(-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']))) \
        - p['Wc_a']*x[0]*2**(0.1*x[2]-2.5), \
        1/p['CO2cap']*(-((1-np.exp(-p['laiW']*x[0]))*p['photI0']*d[0]* \
        (-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']) \
        /(p['photI0']*d[0]+(-p['photCO2_1']*x[2]**2+p['photCO2_2']*x[2]-p['photCO2_3'])*(x[1]-p['photGamma']))) \
        + p['CO2c_a']*x[0]*2**(0.1*x[2]-2.5) + u[0]/1e6 - (u[1]/1e3+p['leak'])*(x[1]-d[1])),        
        1/p['aCap']*(u[2] - (p['ventCap']*u[1]/1e3+p['trans_g_o'])*(x[2]-d[2]) + p['rad_o_g']*d[0]),        
        1/p['H2Ocap']*((1-np.exp(-p['laiW']*x[0]))*p['evap_c_a']*(p['satH2O1']/(p['R']*(x[2]+p['T']))* \
        np.exp(p['satH2O2']*x[2]/(x[2]+p['satH2O3']))-x[3]) - (u[1]/1e3+p['leak'])*(x[3]-d[3]))])    

    return ki
    
def ghat(x,u,d,p,h):
    
    y = np.array([1e3*x[0],1e-3*co2dens2ppm(x[2],x[1]), x[2], vaporDens2rh(x[2], x[3])])
         
    return y

def initMpc(p,ops):
      
    # NMPC prediction horizon (#time steps)
    ops['Np']       = math.floor(6*3600/ops['h'])
    ops['Np_hours'] = ops['Np']*ops['h']/60/60
    
    # Controller bounds: u = [co2, vent, heat]
    # parameter 					        description 									           [unit] 		nominal value
    co2SupplyMin = 0                        # lower bound on CO2 supply rate                [kg{co2} m^{-2} s^{-1}] 0
    co2SupplyMax = 1.2e-6*1e6               # upper bound on CO2 supply rate                [mg{co2} m^{-2} s^{-1}] 1.2e-6
    ventMin = 0                             # lower bound on ventilation rate               [m s^{-1}]              0
    ventMax = 7.5e-3*1e3                    # upper bound on ventilation rate               [mm s^{-1}]             7.5e-3
    heatMin = 0                             # lower bound on energy input through heating   [W m^{-2}]              0
    heatMax = 150                           # upper bound on energy input through heating   [W m^{-2}]              150
    
    # state constraints: x = [dw, co2In, tempIn, vapIn]
    # parameter                             description 									        [unit] 		   nominal value
    WMin = 0                                # lower bound on dry weight                    [kg{crop} m^{-2}]        
    WMax = float('inf')                    # upper bound on dry weight                     [kg{crop} m^{-2}]        
    co2Min = 0                             # lower bound on CO2 concentration              [kg{co2} m^{-3}]        0
    co2Max = 2.75e-3                       # upper bound on CO2 concentration              [kg{co2} m^{-3}]        2.75e-3
    tempMin = 6.5                          # lower bound on temperature                    [°C]                    6.5
    tempMax = 40                           # upper bound on temperature                    [°C]                    40
    rhMin = 0                              # lower bound on relative humidity              [#]                     0
    rhMax = 70                             # upper bound on relative humidity              [#]                     90
    ahMax = rh2vaporDens(tempMax,rhMax)    # upper bound on vapor density                  [kg/m^{-3}]                     
    
    # state constraints vector
    ops['x_min']   = np.array([WMin, co2Min, tempMin, rhMin])
    ops['x_max']   = np.array([WMax, co2Max, tempMax, ahMax])
    
    # output constraints vector
    ops['y_min']   = np.array([WMin, co2Min, tempMin, rhMin])
    ops['y_max']   = np.array([WMax, co2dens2ppm(tempMax,co2Max)/1e3, tempMax, vaporDens2rh(tempMax,ahMax)])
    ops['y_min_k'] = []
    ops['y_max_k'] = []
    
    # control input constraints vector
    ops['u_min']   = np.array([co2SupplyMin, ventMin, heatMin])
    ops['u_max']   = np.array([co2SupplyMax, ventMax, heatMax])
    
    # lower and upper bound change of u 
    ops['du_max']  = np.divide(ops['u_max'],[10,10,10])   

    # initial values of the decision variables (control signals) in the optimization
    ops['u0']      = np.matrix.flatten(np.zeros((ops['nu'],ops['Np'])))
    
    
    # us = [ u1_0 u1_1 .. u1_Np-1  
    #        u2_0 u2_1 .. u2_Np-1  
    #        u3_0 u3_1 .. u3_Np-1 ...], with ui_j = ui(j)
    
    # Us = [u1(0) u1(1) .. u1(Np-1) u2(0) u2(1) ... u2(Np-1)]
       
    x0  = casadi.SX.sym('x0',ops['nx'],1)
    us  = casadi.SX.sym('us',ops['nu'],ops['Np'])
    dus = casadi.diff(us,1,1)
    Us  = casadi.transpose(us)[:] 
    ds  = casadi.SX.sym('ds',ops['nd'],ops['Np'])
    xs  = casadi.SX.sym('xs',ops['nx'],ops['Np']+1)
    ys  = casadi.SX.sym('ys',ops['nx'],ops['Np'])
    
    xs[:,0] = x0
    for ll in range(ops['Np']):
        xs[:,ll+1] = fdhat(xs[:,ll], us[:,ll], ds[:,ll], p, ops['h'])
        ys[:,ll]   = ghat(xs[:,ll], us[:,ll], ds[:,ll], p, ops['h'])
    
    # choose here a cost Js, 
    # (non)linear inequality contraints cs < 0 and 
    # (non)linear equality contraints ceqs = 0
    Js              = -1e3*ys[0,-1] + 10*casadi.sum2(us[0,:])+ \
                        casadi.sum2(us[1,:])+ casadi.sum2(us[2,:])            
    cs              = casadi.horzcat(ys[1,1:]-ops['y_max'][1], \
                        ys[2,1:]-ops['y_max'][2], \
                        ys[3,1:]-ops['y_max'][3], \
                        -ys[0,1:]+ops['y_min'][0], \
                        -ys[1,1:]+ops['y_min'][1], \
                        -ys[2,1:]+ops['y_min'][2], \
                        -ys[3,1:]+ops['y_min'][3])

    nc              = cs.shape[1]                          # number of inequality constraints
    ss              = casadi.SX.sym('ss',nc,1)             # one slack per constraint
    cs              = cs - ss.T                            # softened constraints (cs is a row)
    Js              = Js + 1e5*casadi.sumsqr(ss)           # penalty on the constraint violation
    
    ceqs            = casadi.horzcat([],[])#casadi.horzcat(ys[0,-1]-2, ys[1,-1]-1)                    

    ops['Fs']       = casadi.Function('Fs',[x0,Us,ds,ss],[Js,cs,ceqs],['x0','Us','ds','ss'],['J','c','ceq'])    
    ops['us']       = us
    ops['Us']       = Us
    ops['dus']      = dus
    ops['ds']       = ds
    ops['x0']       = x0
    ops['ys']       = ys   
    ops['Js']       = Js
    ops['ss']       = ss
    ops['nc']       = nc
    
    # bounds on the control signals
    ops['lbu'] = []
    ops['ubu'] = []
    for ll in range(ops['nu']):
        ops['lbu'] = np.append(ops['lbu'],ops['u_min'][ll]*np.ones((ops['Np'],1)))
        ops['ubu'] = np.append(ops['ubu'],ops['u_max'][ll]*np.ones((ops['Np'],1)))

    # bounds on the slack variables
    ops['lbs'] = np.zeros(nc)
    ops['ubs'] = math.inf*np.ones(nc)
    
    ops['lbg'] = np.append(-math.inf*np.ones((np.size(cs,1),1)),np.zeros((np.size(ceqs,1),1)))
    ops['ubg'] = np.append(np.zeros((np.size(cs,1),1)),np.zeros((np.size(ceqs,1),1)))
    
    for ll in range(ops['nu']):
        ops['lbg']   = np.append(ops['lbg'] , -ops['du_max'][ll]*np.ones((ops['Np'],1)))
        ops['ubg']   = np.append(ops['ubg'] , ops['du_max'][ll]*np.ones((ops['Np'],1)))

    return ops  

def updateConstraint(x,u,d,p,ops):
               
    ds  = ops['ds']
    Us  = ops['Us']
    x0  = ops['x0']
    ys  = ops['ys']   
    Js  = ops['Js']
    ss  = ops['ss']
    
    if d[0]<10: # during night, lower the temperature in the greenhouse
        ops['y_min'][2] = 10
        ops['y_max'][2] = 15
    else:
        ops['y_min'][2] = 15
        ops['y_max'][2] = 20

    # store the constraint in time-varying contraint vector
    ops['y_min_k']  = np.append(ops['y_min_k'], ops['y_min'])
    ops['y_max_k']  = np.append(ops['y_max_k'], ops['y_max'])
    
    # update constraints used in the optimization
    cs              = casadi.horzcat(ys[1,1:]-ops['y_max'][1], \
                        ys[2,1:]-ops['y_max'][2], \
                        ys[3,1:]-ops['y_max'][3], \
                        -ys[0,1:]+ops['y_min'][0], \
                        -ys[1,1:]+ops['y_min'][1], \
                        -ys[2,1:]+ops['y_min'][2], \
                        -ys[3,1:]+ops['y_min'][3])                                  
                                  
    cs              = cs - ss.T
        
    ceqs            = casadi.horzcat([],[])                    
    
    ops['Fs']       = casadi.Function('Fs',[x0,Us,ds,ss],[Js,cs,ceqs],['x0','Us','ds','ss'],['J','c','ceq'])    
    
    # update linear inequality constraint
    ops['dUs']      = casadi.transpose(casadi.horzcat(ops['us'][:,0]-u,ops['dus'])) 
    ops['dUs']      = ops['dUs'][:]
    
    return ops

def controller(x0,d,p,ops):
      
    u0      = ops['u0'] 
    g       = []
    lbg     = []
    ubg     = []
    cost    = 0.0
    
    sigmad  = .0
    Ns      = 1
  
    for ll in range(Ns):
        
            dmax  = (1+sigmad)*d
            dmin  = (1-sigmad)*d 
            di    = np.random.uniform(low=dmin, high=dmax, size=(ops['nd'],ops['Np']))
        
            [costi,gi,lbu,ubu,lbgi,ubgi] = evaluateCostConstraints(x0,di,p,ops)
            
            lbg   = casadi.vertcat(lbg,lbgi)
            ubg   = casadi.vertcat(ubg,ubgi)

            g     = casadi.horzcat(g,gi)
            cost  = cost + costi
            
    cost = cost/Ns
   
    # decision vector = controls and slacks
    nU      = ops['nu']*ops['Np']
    X       = casadi.vertcat(ops['Us'],ops['ss'])
    lbx     = np.append(lbu,ops['lbs'])
    ubx     = np.append(ubu,ops['ubs'])
    x0g     = np.append(u0,np.zeros(ops['nc']))      # slacks start at zero
   
    prob    = {'f':cost,'x':X,'g':g}
    solver  = casadi.nlpsol('solver','ipopt',prob,{'ipopt': {'warm_start_init_point': 'yes', 'print_level':3}}) 
     
    if 'lam_x0' in ops:
        output        = solver(x0=x0g, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_x0=ops['lam_x0'], lam_g0=ops['lam_g0'])
    else:
        output        = solver(x0=x0g, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        ops['lam_g0'] = output['lam_g']
        ops['lam_x0'] = output['lam_x']
        
    Xopt        = output['x'].full().flatten()
    Uopt        = Xopt[:nU] # controls
    ops['sopt'] = Xopt[nU:] # slacks (> 0 means a constraint is violated)
    ops['u0']   = Uopt
    uopt        = np.reshape(Uopt,(ops['nu'],ops['Np']))

    return uopt[:,0]

def evaluateCostConstraints(x0,d,p,ops):
               
    temp  = ops['Fs'].call({'x0':x0,'Us':ops['Us'],'ds':d,'ss':ops['ss']})
                
    cost  = temp['J']
    c     = temp['c']
    ceq   = temp['ceq']
                
    lbu   = ops['lbu']
    ubu   = ops['ubu']
                
    g     = casadi.horzcat(c, ceq, casadi.transpose(ops['dUs']))
    lbg   = ops['lbg']
    ubg   = ops['ubg']
                
    return cost,g,lbu,ubu,lbg,ubg

###############################################################################
###############################################################################

               ####### plot functions ############

###############################################################################
###############################################################################
def plotTrainingData(D,U,X,Y,T,ops):
 
    c        = 86400 
    
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
        "training data",
        fontsize=23,
        y=0.99
    )
    
    # Plot all datasets
    for nn in range(len(D)):
    
        t = T[nn]
        d = D[nn].T
        u = U[nn].T
        y = Y[nn].T
    
        # Left column: disturbances d, followed by inputs u
        for ll in range(ops['nd']):
            a = ax[ll, 0]
            a.plot(t/c, d[ll, :], drawstyle='steps')
            a.set_ylabel(d_labels[ll], fontsize=18)
            a.grid(True)
    
        for ll in range(ops['nu']):
            row = ops['nd'] + ll
            a = ax[row, 0]
            a.plot(t/c, u[ll, :], drawstyle='steps')
            a.set_ylabel(u_labels[ll], fontsize=18)
            a.grid(True)
    
        # Right column: outputs y
        for ll in range(ops['nx']):
            a = ax[ll, 1]
            a.plot(t/c, y[ll, :], drawstyle='steps')
            a.set_ylabel(y_labels[ll], fontsize=18)
            a.grid(True)
    
    # y-limits from the extremes over ALL datasets (set once, after the loop)
    def ylims(arrs, ll):
        lo = min(np.nanmin(A[:, ll]) for A in arrs)
        hi = max(np.nanmax(A[:, ll]) for A in arrs)
        lo = 1.1*lo if lo < 0 else 0.0      # zero baseline, unless the data go negative
        hi = 1.1*hi if hi > 0 else 0.0
        if hi <= lo:
            hi = lo + 1.0                   # avoid a degenerate (flat) range
        return lo, hi
    
    for ll in range(ops['nd']):
        ax[ll, 0].set_ylim(*ylims(D, ll))
    
    for ll in range(ops['nu']):
        ax[ops['nd'] + ll, 0].set_ylim(*ylims(U, ll))
    
    for ll in range(ops['nx']):
        if ll == 3:
            ax[ll, 1].set_ylim(0, 100)
        else:
            ax[ll, 1].set_ylim(*ylims(Y, ll))
    
    # Hide unused panels
    for ll in range(nrows):
        if ll >= n_left:
            ax[ll, 0].set_visible(False)
        if ll >= ops['nx']:
            ax[ll, 1].set_visible(False)
    
    # X-axis and labels (limits over all datasets)
    tmin = min(t[0] for t in T) / c
    tmax = max(t[-1] for t in T) / c
    
    for col in range(2):
        visible_axes = [ax[ll, col] for ll in range(nrows)
                        if ax[ll, col].get_visible()]    
        for a in visible_axes:
            a.set_xlim(tmin, tmax)
    
        if visible_axes:
            visible_axes[-1].set_xlabel('Time (days)', fontsize=18)
    
    ax[ops['nx'] - 1, 1].tick_params(axis='x', labelbottom=True)
    
    # Reduce space above the first row
    fig.tight_layout(rect=[0, 0, 1, 0.985], pad=0.5, h_pad=0.5, w_pad=2)

    #plt.savefig("trainScenarios.svg", format="svg")
    #plt.close()

def plotTrainingValidation(d,u,x,y,t,yhat,ops):
 
    c        = 86400 
    
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
        "comparison of one random scenario from training data: model (red) and data (blue)",
        fontsize=23,
        y=0.99
    )
    
    # Left column: disturbances d, followed by inputs u
    for ll in range(ops['nd']):
        a = ax[ll, 0]
        a.plot(t / c, d[ll, :], drawstyle='steps')
        a.set_ylabel(d_labels[ll], fontsize=18)
        a.grid(True)
    
    for ll in range(ops['nu']):
        row = ops['nd'] + ll
        a = ax[row, 0]
        a.plot(t / c, u[ll, :], drawstyle='steps')
        a.set_ylabel(u_labels[ll], fontsize=18)
        a.grid(True)
    
    # Right column: outputs y
    for ll in range(ops['nx']):
        a = ax[ll, 1]
        a.plot(t / c, y[ll, :], drawstyle='steps')
        a.plot(t / c, yhat[ll, :], 'r--', drawstyle='steps')
        a.set_ylabel(y_labels[ll], fontsize=18)
        a.grid(True)
    
        if ll == 3:
            a.set_ylim(0, 100)
        else:
            ymax = max(np.nanmax(y[ll, :]), np.nanmax(yhat[ll, :]))   # max over data and model
            a.set_ylim(bottom=0, top=1.1 * ymax if ymax > 0 else 1.0)    
    
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
            a.set_xlim(t[0] / c, t[-1] / c)
    
        if visible_axes:
            visible_axes[-1].set_xlabel('Time (days)', fontsize=18)
    
    ax[ops['nx'] - 1, 1].tick_params(axis='x', labelbottom=True)
    
    # Reduce space above the first row
    fig.tight_layout(rect=[0, 0, 1, 0.985], pad=0.5, h_pad=0.5, w_pad=2)

    #plt.savefig("trainValidation.svg", format="svg")
    #plt.close()
    
def plotTestValidation(d,u,x,y,t,yhat,ops):
 
    c        = 86400 
    
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
        "comparison of scenario from test data: model (red) and data (blue)",
        fontsize=23,
        y=0.99
    )
    
    # Left column: disturbances d, followed by inputs u
    for ll in range(ops['nd']):
        a = ax[ll, 0]
        a.plot(t / c, d[ll, :], drawstyle='steps')
        a.set_ylabel(d_labels[ll], fontsize=18)
        a.grid(True)
    
    for ll in range(ops['nu']):
        row = ops['nd'] + ll
        a = ax[row, 0]
        a.plot(t / c, u[ll, :], drawstyle='steps')
        a.set_ylabel(u_labels[ll], fontsize=18)
        a.grid(True)
    
    # Right column: outputs y
    for ll in range(ops['nx']):
        a = ax[ll, 1]
        a.plot(t / c, y[ll, :], drawstyle='steps')
        a.plot(t / c, yhat[ll, :], 'r--', drawstyle='steps')
        a.set_ylabel(y_labels[ll], fontsize=18)
        a.grid(True)
    
        if ll == 3:
            a.set_ylim(0, 100)
        else:
            ymax = max(np.nanmax(y[ll, :]), np.nanmax(yhat[ll, :]))   # max over data and model
            a.set_ylim(bottom=0, top=1.1 * ymax if ymax > 0 else 1.0)    
    
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
            a.set_xlim(t[0] / c, t[-1] / c)
    
        if visible_axes:
            visible_axes[-1].set_xlabel('Time (days)', fontsize=18)
    
    ax[ops['nx'] - 1, 1].tick_params(axis='x', labelbottom=True)
    
    # Reduce space above the first row
    fig.tight_layout(rect=[0, 0, 1, 0.985], pad=0.5, h_pad=0.5, w_pad=2)
    
    #plt.savefig("TestValidation.svg", format="svg")
    #plt.close()