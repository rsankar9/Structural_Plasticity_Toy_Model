

import ArmActivity
import numpy as np
import matplotlib.pyplot as plt
import os

if os.path.isdir('Results_Trial') == False:
    os.mkdir('Results_Trial')

f = 0
Nruns = 5

R_RL_HL = np.zeros(Nruns)

for run in range(Nruns):
    f = f + 1
    rSeed = np.random.randint(0, 1e7)
    resFile = 'Results_Trial/RL_HL_' + str(f) + '_trial'
    r_RL_HL = ArmActivity.SPTModel(resFile, rSeed)

    R_RL_HL[run] = r_RL_HL
    
fig = plt.figure()
x = np.arange(1, Nruns+1)
plt.plot(x, R_RL_HL, label='RL HL')
plt.xlabel('Simulation #')
plt.ylabel('Reward')
plt.ylim(top=1, bottom=0)
plt.legend()
plt.title('Mean reward over last 5% trials of each simulation')
plt.savefig('Results_Scaled/OverallResults.png')
#plt.close()