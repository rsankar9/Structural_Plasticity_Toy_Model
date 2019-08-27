#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:38:18 2019

@author: rsankar

Written to test the 1-segmented arm case only.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def gaussian(x, mu, sig):
    """
    Compute gaussian distribution. Not used anymore.
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def SPTModel(arg_resFile, arg_rSeed):
    """
    Simulate arm exploration.
    """
    np.random.seed(arg_rSeed)
    
    # Arm constraints
    n_arms = 1  
    lengths = np.ones((n_arms))
    total_l = np.sum(lengths)
    min_a, max_a = 0, 2*np.pi 
          
    # Activity parameters
    RL_mean, RL_std = np.pi/2, 0.8
    HL_mean, HL_std = 0, 0.1
    rho_RL, rho_HL = 1, 2                                                      # Attenuation factor
    
    # Target
    output_range = 2 * total_l
#    theta = np.random.uniform(min_a, max_a)                                        # UNCOMMENT.
    theta = np.random.uniform(0, np.pi)                                         # REMOVE. Only to test.
    target = np.array([np.cos(theta), np.sin(theta)]) * lengths
    
    # Simulation parameters
    ntrials = 3000
    HL, RL = 1, 1
    noise_std = 0.15
    
    # Updation parameters
    eta_HL, eta_RL = 0.005, 0.35

    R = np.zeros(ntrials)
    E = np.zeros(ntrials)
    positions = np.zeros((ntrials, 2))
    RL_angles = np.zeros((ntrials))
    HL_angles = np.zeros((ntrials))
    T_angles = np.zeros((ntrials))

    
    # Simulating
    for nt in range(ntrials):
        # Calculating LMAN and HVc activities
        angles_x = np.linspace(min_a, max_a, 10000)
        activity_RL = scipy.stats.norm.pdf(angles_x, RL_mean, RL_std) / rho_RL * RL 
        activity_HL = scipy.stats.norm.pdf(angles_x, HL_mean, HL_std) / rho_HL * RL 
        activity_T = activity_RL + activity_HL
        noise = np.random.normal(0, noise_std)
        angles_t = np.average(angles_x, weights=activity_T) + noise
        if nt%1000==0:
            print('noise:', noise, 'avg:', np.average(angles_x, weights=activity_T), 'RL_mean:', RL_mean)

        output = lengths * np.cos(angles_t), lengths * np.sin(angles_t)
        
        # Keeping track for plotting
        positions[nt] = output
        RL_angles[nt] = RL_mean
        HL_angles[nt] = HL_mean
        T_angles[nt] = angles_t

        # Calculating error and reward
        error = np.sqrt((output - target) ** 2).sum()
        E[nt] = error / (output_range ** 2)
        R[nt] = np.exp(-E[nt] ** 2 / 0.55 ** 2)
        
         # Updation of HL angles
        if HL:
            angles_hl = HL_mean
            d_angle_hl = eta_HL * (angles_t - angles_hl) * HL
            angles_hl += d_angle_hl * HL
            angles_hl = angles_hl % max_a
            HL_mean = angles_hl

        # Updation of RL angles
        if RL and nt > 25:
            angles_rl = RL_mean
            R_prev = R[nt - 25: nt].sum() / 25                                      # Mean reward over past n trials 
            d_angle_rl = eta_RL * (R[nt] - R_prev) * noise * RL
            angles_rl += d_angle_rl * RL
            angles_rl = angles_rl % max_a
            RL_mean = angles_rl
            if nt%1000==0:
                print('R_diff:', R[nt] - R_prev, 'eta_RL:', eta_RL, 'noise:', noise, 'd_angle_rl:', d_angle_rl, 'RL_mean:', RL_mean)
            
        # Updation of attenuation factor
        ### Increasing / decreasing rho_RL or rho_HL.

            
    print('Target : angle=', theta, ', position=', target)
    print('Final output:', output)
    print('Final reward:', R[-1])
    print('Final error:', E[-1])
    print('Final angles RL:', RL_mean)
    print('Final angles HL:', HL_mean)
    print('Final angles Total:', angles_t)
    
    print('----Plotting----')
    
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.6)
    
    # Plotting reward and error
    ax = fig.add_subplot(4, 2, (1, 2))
    
    x = np.arange(ntrials)
    ax.plot(x, E, label='Error', marker=',', color='orange')
    ax.plot(x, R, label='Reward', marker=',', color='green')
    ax.set_ylim(top=1, bottom=0)
    ax.set_xlabel('# of Trials')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Reward and normalised error over 5 trials')
    ax.legend()

    # Plotting angles after each trial
    ax = fig.add_subplot(4, 2, (3, 4))

    x = np.arange(ntrials)
    ax.plot(x, T_angles, label='Total arm', marker='.', linewidth=0, color='blue', alpha=0.2)
    ax.plot(x, RL_angles, label='RL arm', linestyle=':', color='orange')
    ax.plot(x, HL_angles, label='HL arm', linestyle=':', color='green')
    ax.plot(x, theta * np.ones(len(x)), label='Target', linestyle='-', color='black')
    ax.set_ylim(top=2 * np.pi, bottom=0)
    ax.set_xlabel('# of Trials')
    ax.set_ylabel('Angles')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('RL, HL and total angles every trial')
    ax.legend(bbox_to_anchor=(0, -1.2), loc='lower left', borderaxespad=0.)

    # Plotting position after every trials
    ax = fig.add_subplot(4, 2, (5, 8))
    
    r = np.linspace(0,2*np.pi,1000)                                                 # Plotting outer circle
    x, y = np.cos(r) * total_l, np.sin(r) * total_l
    ax.plot(x,y,linestyle=':',label='Reach')                                
#    ax.plot(positions[:,0], positions[:,1], marker='o', linewidth=0, color = 'orange', alpha = 0.2)
    ax.plot(positions[int(0.95*ntrials):,0], positions[int(0.95*ntrials):,1], marker='o', linewidth=0, color = 'orange', alpha = 1)
#    for nt in range(ntrials):
#        ax.plot(positions[nt,0], positions[nt,1], marker='o', linewidth=0, color = 'orange', alpha = 1/(ntrials*ntrials) * nt * nt)
    ax.plot(0,0,color='black',marker='o',label='Origin')                            # Plotting (0,0)
    ax.plot(target[0], target[1],color='green',marker='X',label='Target')           # Plotting target point
    
    ax.set_ylim((-total_l,total_l))
    ax.set_xlim((-total_l,total_l))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(1.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Output and farthest arm at the last few timesteps')
    ax.legend(bbox_to_anchor=(1.5, 0), loc='lower right', borderaxespad=0.)
    
    plt.savefig(arg_resFile)
#    plt.close()
    
    return np.average(R[int(0.95*ntrials):])                                    # Returns mean reward over last 5% trials.
    

    
        
