import numpy as np
import HL_solely
import RL_solely
import early_HL_RL
import late_HL_RL
import gc
import matplotlib.pyplot as plt

test = 1

if test == 1:

    f = 0
    R = np.array([])
    RT = np.array([])
    Seq_IN = np.array([])
    Seq_OUT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_HL_solely_' + str(f)
        r_learn, r_test, seq_in, seq_out = HL_solely.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)
        Seq_IN = np.append(Seq_IN, seq_in)
        Seq_OUT = np.append(Seq_OUT, seq_out)

    print(Seq_IN.ravel(), '\n', Seq_OUT.ravel())
    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Simulation #')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)

    for i in n:
        plt.text(i, 0.5, str(Seq_IN.ravel()[i] + '\n|\n' + Seq_OUT.ravel()[i]))

    Title = str((R >= 0.9).sum()) + '/' + str(len(R)) + 'simulations ended with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result_HL_solely.png')
    plt.close()



# elif test == 2:

    f = 0
    R = np.array([])
    RT = np.array([])
    Seq_IN = np.array([])
    Seq_OUT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_RL_solely_' + str(f)
        r_learn, r_test, seq_in, seq_out = RL_solely.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)
        Seq_IN = np.append(Seq_IN, seq_in)
        Seq_OUT = np.append(Seq_OUT, seq_out)

    print(Seq_IN.ravel(), '\n', Seq_OUT.ravel())
    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Simulation #')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)

    for i in n:
        plt.text(i, 0.5, str(Seq_IN.ravel()[i] + '\n|\n' + Seq_OUT.ravel()[i]))

    Title = str((R >= 0.9).sum()) + '/' + str(len(R)) + 'simulations ended with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result_RL_solely.png')
    plt.close()


# elif test == 3:

    f = 0
    R = np.array([])
    RT = np.array([])
    Seq_IN = np.array([])
    Seq_OUT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_early_HL_RL_' + str(f)
        r_learn, r_test, seq_in, seq_out = early_HL_RL.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)
        Seq_IN = np.append(Seq_IN, seq_in)
        Seq_OUT = np.append(Seq_OUT, seq_out)

    print(Seq_IN.ravel(), '\n', Seq_OUT.ravel())
    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Simulation #')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)

    for i in n:
        plt.text(i, 0.5, str(Seq_IN.ravel()[i] + '\n|\n' + Seq_OUT.ravel()[i]))

    Title = str((R >= 0.9).sum()) + '/' + str(len(R)) + 'simulations ended with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result_early_HL_RL.png')
    plt.close()



# elif test == 4:

    f = 0
    R = np.array([])
    RT = np.array([])
    Seq_IN = np.array([])
    Seq_OUT = np.array([])
    Nruns = 10
    for run in range(Nruns):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_late_HL_RL_' + str(f)
        r_learn, r_test, seq_in, seq_out = late_HL_RL.SPTModel(resFile, rSeed)
        R = np.append(R, r_learn)
        RT = np.append(RT, r_test)
        Seq_IN = np.append(Seq_IN, seq_in)
        Seq_OUT = np.append(Seq_OUT, seq_out)

    print(Seq_IN.ravel(), '\n', Seq_OUT.ravel())
    # Plotting metrics
    n = range(len(R))
    plt.plot(n, R, label='Learning')
    plt.plot(n, RT, label='Testing')
    plt.xlabel('Simulation #')
    plt.ylabel('Mean reward over last few trials')
    plt.ylim(0, 1)

    for i in n:
        plt.text(i, 0.5, str(Seq_IN.ravel()[i] + '\n|\n' + Seq_OUT.ravel()[i]))

    Title = str((R >= 0.9).sum()) + '/' + str(len(R)) + 'simulations ended with >90% reward.'
    plt.title(Title)
    plt.legend()
    plt.savefig('Results/Overall_Result_late_HL_RL.png')
    plt.close()

