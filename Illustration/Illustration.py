#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:04:56 2020
@author: rsankar

This script tries to illustrate the idea that 2 pathways can work in tandem where one pathway acts as a tutor signal to the other pathway.
The illustration uses the combination of 2 sine waves of different frequencies as a target.
One pathway, based on RL, tries to reach the right amplitude of the faster frequency sine wave.
The second pathway, uses RL as a tutor signal, and mimics it without any other access to the goal.
Meanwhile, akin to the the connections between HVc-RA sprouting over time, the amplitude of the slower frequency sine wave increases.
"""

import numpy as np
import matplotlib.pyplot as plt

### Initialisations
seed = np.random.randint(1e8)
np.random.seed(seed)

# Setting - Delayed pathway 1 (True) or not (False).
delayed = True

# Parameters
N = 500                                                                 # No. of points in curve
ntrials = 40000

# Target wave is combination of 2 sine waves of different frequencies
t = np.linspace(0,4*np.pi,N)
wav1 = 10*np.sin(t)
wav2 = np.sin(10*t)
target = wav1 + wav2

# Variables to learn (Amplitudes of wav1 and wav2)
rl = 0.00                                                               # Output of pathway 2 and amplitude of wav2
hl1 = 0.00                                                              # Independent and amplitude of wav1
hl2 = 0.00                                                              # Output of pathway 1 and amplitude of wav2

# To keep track
R_prev = 0
Rs = np.zeros(ntrials)
rls = np.zeros(ntrials)
hl1s = np.zeros(ntrials)
hl2s = np.zeros(ntrials)

if delayed == False:
    figname = 'Result_illustration'
else:
    figname = 'Result_illustration_delayed'

### Learning
for i in range(ntrials):
    
    # Output
    noise = np.random.uniform(-1,1) * 0.5
    RL_output = (rl+noise)*np.sin(10*t)                                 # Exploration of pathway 2
    HL_output = hl1 * np.sin(t) + hl2*np.sin(10*t)                      # Output of pathway 1
    T_output = RL_output + HL_output
    
    # Error and reward
    error = np.sqrt(np.sum(np.abs((T_output - target)) ** 2)) / (N)
    R = np.exp(-error**2/0.4**2)

    # Updation of pathway 1 as per reward
    drl = 0.3 * (rl + noise) * (R-R_prev)
    rl = rl + drl
    
    # Updation of pathway 2 using the tutor signal (unrealistic)
    if delayed == False or i>5000:
        dhl2 = 0.03 * rl * np.sign(drl)
        hl2 += dhl2

    # Independent growth of amplitude of wav1 (unrealistic)
    hl1 += 0.0001 * (10-hl1)

    # Controlling for negative values
    rl = np.maximum(0, rl)
    hl1 = np.maximum(0, hl1)
    hl2 = np.maximum(0, hl2)

    # Recently received reward
    if i>20:
        R_prev = np.mean(Rs[i-19:i+1])

    # For plotting
    rls[i] = rl
    hl1s[i] = hl1
    hl2s[i] = hl2
    Rs[i] = R



### Plotting
fig,ax = plt.subplots(3)

ax[0].plot(t/np.pi,RL_output, label='rl')
ax[0].plot(t/np.pi,HL_output, label='hl')
ax[0].plot(t/np.pi,T_output, label='T')
ax[0].plot(t/np.pi,target,'red', label='target',alpha=0.2)
ax[0].legend()
ax[0].set_title('Final output')

ax[1].plot(rls, label='rl')
ax[1].plot(hl2s, label='hl2')
ax[1].plot(hl1s, label='hl1')
ax[1].legend(loc='right')
ax[1].set_title('Amplitude')


ax[2].plot(Rs, label='reward')
ax[2].legend()
ax[2].set_title('Reward')

plt.savefig(figname+'.png')
plt.show()
