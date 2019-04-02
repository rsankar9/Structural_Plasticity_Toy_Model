"""
Using n-arm exploration to understand the effect of RL aiding HL
"""

import numpy as np
import matplotlib.pyplot as plt
   
def SPTModel(arg_resFile, arg_rSeed, arg_HL, arg_early):
    np.random.seed(arg_rSeed)  
    
    n_arms = 2                                                                      # Arm parameters
    min_a, max_a = 0, 2*np.pi
    lengths = np.array([1,1])
    total_l = np.sum(lengths)
    pos = np.zeros((n_arms, 2))
    output_range = 2 * total_l                                                      # Maximum error possible
    
    ntrials = 30000                                                                 # Simulation parameters
    theta = np.random.uniform(min_a, max_a)
    l = np.random.uniform(0, total_l)
    target = np.array([l * np.cos(theta), l * np.sin(theta)])
    
    R = np.zeros(ntrials)
    E = np.zeros(ntrials)
    positions = np.zeros((ntrials, n_arms, 2))
    RL_angles = np.zeros((ntrials, n_arms))
    HL_angles = np.zeros((ntrials, n_arms))
    T_angles = np.zeros((ntrials, n_arms))

    HL, RL = arg_early, 1

    angles_rl = np.random.uniform(min_a, max_a, n_arms)
    angles_hl = np.zeros(n_arms)                                                               # Learning type
    
    
    print('----Simulations running----')
    
    for nt in range(ntrials):
        if nt > ntrials/3: HL = arg_HL
    
        # Separate arrays for clarity; reduce later
        noise = np.random.normal(0, 0.1, 2)
        angles_t = (angles_hl * HL + angles_rl * RL + noise * RL) % max_a
        
        # Calculating position of arm
        pos[:,0], pos[:,1] = lengths * np.cos(angles_t), lengths * np.sin(angles_t)
        pos[...] = np.cumsum(pos, axis=0)
        output = pos[-1]                                                            # Position of final arm
    
        # Calculating error and reward
        error = np.sqrt(((output - target) ** 2).sum())
        E[nt] = error / output_range
        R[nt] = np.exp(-E[nt] ** 2 / 0.55 ** 2)
    
        # Updation of angles
        d_angle_hl = 0.0001 * (angles_t - angles_hl) * HL
        angles_hl[1] += d_angle_hl[1] * HL
        angles_hl = angles_hl % max_a
        
        R_prev = 0
        d_angle_rl = [0,0]
        if nt >= 25:
            R_prev = R[nt - 25: nt].sum() / 25                                      # Mean reward over past n trials 
            d_angle_rl = 0.05 * (R[nt] - R_prev) * noise * RL
            angles_rl += d_angle_rl * RL
            angles_rl = angles_rl % max_a
        
        # For keeping track
        positions[nt] = pos
        RL_angles[nt] = angles_rl
        HL_angles[nt] = angles_hl
        T_angles[nt] = angles_t

    
    print('Final output:', output)
    print('Final reward:', R[-1])
    print('Final error:', E[-1])
    print('Final angles RL:', angles_rl)
    print('Final angles HL:', angles_hl)
    
    print('----Plotting----')
    
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.6)
    
    # Plotting reward and error, averaged over 100 trials
    ax = fig.add_subplot(4, 2, (1, 2))
    
    x = np.arange(ntrials-100+1)
    conv = np.ones(100)
    R_avgd = np.convolve(R, conv, 'valid')/100
    E_avgd = np.convolve(E, conv, 'valid')/100
    ax.plot(x, E_avgd, label='Error', marker=',', color='orange')
    ax.plot(x, R_avgd, label='Reward', marker=',', color='green')
    ax.plot(x, E[99:], alpha=0.1, marker=',', color='orange')
    ax.plot(x, R[99:], alpha=0.1, marker=',', color='green')
    
    ax.set_ylim(top=1, bottom=0)
    ax.set_xlabel('# of Trials')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Averaged reward and normalised error over 100 trials')
    ax.legend()

    # Plotting angles after each trial
    ax = fig.add_subplot(4, 2, (3, 4))

    x = np.arange(0,ntrials,10)
    ax.plot(x, T_angles[::10, 0], label='Total 1st arm', marker=',', linewidth=0, color='blue', alpha=0.2)
    ax.plot(x, T_angles[::10, 1], label='Total 2nd arm', marker='.', linewidth=0, color='blue', alpha=0.2)
    ax.plot(x, RL_angles[::10, 0], label='RL 1st arm', linestyle='-', color='orange')
    ax.plot(x, RL_angles[::10, 1], label='RL 2nd arm', linestyle=':', color='orange')
    ax.plot(x, HL_angles[::10, 0], label='HL 1st arm', linestyle='-', color='green')
    ax.plot(x, HL_angles[::10, 1], label='HL 2nd arm', linestyle=':', color='green')
    ax.set_ylim(top=2 * np.pi, bottom=0)
    ax.set_xlabel('# of Trials')
    ax.set_ylabel('Angles')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('RL, HL and total angles every 10 trials')
    ax.legend(bbox_to_anchor=(0, -1.2), loc='lower left', borderaxespad=0.)

    # Plotting position after every 100th trials, along with the last arm
    ax = fig.add_subplot(4, 2, (5, 8))
    
    r = np.linspace(0,2*np.pi,1000)                                                 # Plotting outer circle
    x, y = np.cos(r) * total_l, np.sin(r) * total_l
    ax.plot(x,y,linestyle=':',label='Reach')                                
    for nt in range(0,ntrials,100):                                                  # Plotting farthest arms
        ax.plot(positions[nt,:,0], positions[nt,:,1], marker=',', color = 'blue', alpha = nt * nt * 0.000000001111111)
    ax.plot(positions[-1,:,0], positions[-1,:,1], marker=',', color = 'blue', alpha = 1, label='Farthest arm')  # Separately for legend lable
    ax.plot(positions[::100,-1,0], positions[::100,-1,1], marker='.', linewidth=0., label = 'End point', color='orange', alpha = 0.2) # Plotting farthest arm position
    ax.plot(0,0,color='black',marker='o',label='Origin')                            # Plotting (0,0)
    ax.plot(target[0], target[1],color='green',marker='X',label='Target')           # Plotting target point
    
    ax.set_ylim((-2,2))
    ax.set_xlim((-2,2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect(1.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Output and farthest arm at every 100th timestep')
    ax.legend(bbox_to_anchor=(1.5, 0), loc='lower right', borderaxespad=0.)
    
    plt.savefig(arg_resFile)
    plt.close()
    
    return np.mean(R[-500:-1])