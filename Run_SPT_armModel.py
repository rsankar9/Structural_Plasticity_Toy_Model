import ArmExploration
import numpy as np
import matplotlib.pyplot as plt

f = 0
Nruns = 10
R_early = np.zeros(Nruns)
R_late = np.zeros(Nruns)
R_onlyRL = np.zeros(Nruns)
for run in range(Nruns):
    f = f + 1
    rSeed = np.random.randint(0, 1e7)
    resFile = 'Results/RL_HL_' + str(f) + '_earlyHL'
    r_early = ArmExploration.SPTModel(resFile, rSeed, 1, 1)
    resFile = 'Results/RL_HL_' + str(f) + '_lateHL'
    r_late = ArmExploration.SPTModel(resFile, rSeed, 1, 0)
    resFile = 'Results/RL_HL_' + str(f) + '_onlyRL'
    r_onlyRL = ArmExploration.SPTModel(resFile, rSeed, 0, 0)

    R_early[run] = r_early
    R_late[run] = r_late
    R_onlyRL[run] = r_onlyRL

x = np.arange(1, Nruns+1)
plt.plot(x, R_early, label='Early HL')
plt.plot(x, R_late, label='Late HL')
plt.plot(x, R_onlyRL, label='No HL')
plt.xlabel('Simulation #')
plt.ylabel('Reward')
plt.ylim(top=1, bottom=0)
plt.legend()
plt.title('Mean reward over last 500 trials of each simulation')
plt.savefig('Results/OverallResults.png')
plt.close()