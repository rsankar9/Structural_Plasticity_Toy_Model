import numpy as np
import HL_solely
import RL_solely
import late_HL_RL
import early_HL_RL
import gc
import matplotlib.pyplot as plt

test = 0

if test == 1:
    
    f=0
    R = np.array([])
    RT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)
        
        resFile = 'Results/Test_RL_' + str(f)
        r_learn, r_test = RL_solely.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)
    
    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Trials')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)
    Title = 'Reward over trials:' + str((R>=0.9).sum()) + '/' + str(len(R)) + 'with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result.png')
    plt.close()

elif test == 2:

    f=0
    R = np.array([])
    RT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_early_HL_RL_' + str(f)
        r_learn, r_test = early_HL_RL.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)

    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Trials')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)
    Title = 'Reward over trials:' + str((R>=0.9).sum()) + '/' + str(len(R)) + 'with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result.png')
    plt.close()


elif test == 3:

    f=0
    R = np.array([])
    RT = np.array([])
    Nruns = 1
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_late_HL_RL_' + str(f)
        r_learn, r_test = late_HL_RL.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)

    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Trials')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)
    Title = 'Reward over trials:' + str((R>=0.9).sum()) + '/' + str(len(R)) + 'with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result.png')
    plt.close()



elif test == 4:

    f=0
    R = np.array([])
    RT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_HL_' + str(f)
        r_learn, r_test = HL_solely.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)

    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Trials')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)
    Title = 'Reward over trials:' + str((R>=0.9).sum()) + '/' + str(len(R)) + 'with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result.png')
    plt.close()
