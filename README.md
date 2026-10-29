## 21st IFAC Symposium on System Identification: learning models for decision and control

For questions reach out to: sjoerd.boersma@wur.nl

## Data 
The benchmark data can be found in lettuceGreenhouseBenchmarkData.csv and the function plotBenchmarkData.py can be used to load and plot it.

The data in lettuceGreenhouseBenchmarkData.csv contains 5 scenarios:

scenario	| $N_0$| 	date                | Mean temp	 |     Mean radiation
---|---|---|---|---
1 (train)		    | 0	   | 2014-01-10 to 02-18	| 5.6 °C     |     37     
2 (validation)		|20448 | 2014-03-22 to 04-30	| 11.2 °C	 |     161    
3 (train)		    |40896 | 2014-06-01 to 07-10	| 16.2 °C	 |     232    
4 (test) 		    |61344 | 2014-08-11 to 09-19	| 16.0 °C	 |     164    
5 (train)		    |81504 | 2014-10-20 to 11-28	| 10.0 °C	 |     40     

with $N_0$ the starting index of the weather data (disturbance).

## Model 
This directory contains benchmark dataset and a dynamical lettuce greenhouse model that is used to generate the benchmark dataset. This data is for testing system identification methods.

$$
\begin{aligned}
\frac{\text{d}x(t)}{\text{d}t} &= f\left( x(t),u(t),d(t),w(t),p \right), \qquad  &y(t) = g\left( x(t),u(t),d(t),v(t),p \right) \\
\end{aligned}
$$

with continuous time $t \in \mathbb{R}$, nonlinear functions $f(\cdot), g(\cdot)$, model parameter $p$ \in \mathbb{R}^{28}$ and 

variable|meaning|variable|meaning
---|---|---|---
$d(t) \in \mathbb{R}^4$|weather disturbance|$x(t) \in \mathbb{R}^4$|state variable
$u(t) \in \mathbb{R}^3$|control signal|$y(t) \in \mathbb{R}^{4}$|measurement
$w(t)\in \mathbb{R}^4$|process noise | |
$v(t)\in \mathbb{R}^{4}$|measurement noise | |

The content of the variables is defined below.

$d(t)$|$u(t)$|$x(t)$|$y(t)$
---|---|---|---
radiation (W/m2)          | CO2 injection (mg/m2/s) | dry weight (kg/m2)        | dry weight (g/m2)
outdoor CO2 (kg/m3)       | ventilation (mm/s) | indoor CO2 (kg/m3)        | indoor CO2 (ppm)
outdoor temperature (deg) | heating (W/m2) | indoor temperature (deg)  | indoor temperature (g/m2)
outdoor humidity (kg/m3)  | | indoor humidity (kg/m3)   | indoor humidity (%)

For more detail about the model, the reader is referred to:

- Eldert van Henten (1994). Greenhouse climate management: an optimal control approach. Ph.D. thesis, University Wageningen.

## Model Predictive Control
The data generation is done using a model predictive controller. In order to generate new data, install CasADi https://web.casadi.org/get/. 

For more detail about the model predictive controller, the reader is referred to:

- Sjoerd Boersma, Congcong Sun and Simon Van Mourik (2022). Robust sample-based model predictive control of a greenhouse system with parametric uncertainty. IFAC-PapersOnLine, 55 (32) (2022), pp. 177-182.

