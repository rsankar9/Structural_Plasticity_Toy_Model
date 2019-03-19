# HL with RL influence for a single syllable

import numpy as np
import matplotlib.pyplot as plt
import gc

def sigmoid(x, m=5, a=0.5):
    return 1 / (1 + np.exp(-1 * (x - a) * m))

def sliding_window(X, window):
    Y = np.array([X[i:i + 10] for i in range(len(X) - window + 1)])
    return Y


def SPTModel(arg_resFile, arg_rSeed):
    gc.collect()

    # Parameters #
    # ---------- #

    resFile = arg_resFile

    rSeed = arg_rSeed                               # Random seed parameter
    np.random.seed(rSeed)

    HVC_size, RA_size, MO_size = 3, 10, 1           # Structure parameters

    Wepsilon, Wmin_MO, Wmax_MO = 0.05, 0.0, 1.0     # Initialisation parameters

    RA_noise_mean, RA_noise_std = 0.0, 0.00001          # Activity parameters
    MO_noise_mean, MO_noise_std = 0.0, 0.00001

    RA_sig_slope = MO_sig_slope = 5                 # Sigmoidal limits and slope
#    arg_steepFactor = 5

    Hebbian_learning = 0                            # Learning parameters
    pPos, pDec = 0.0005, 0.00005
    Wmin_HL, Wmax_HL = -2.0, 2.0
    Reinforcement_learning = 1
    eta = 0.2
    Wmin_RL, Wmax_RL = -0.2, 0.2
    RLw_noise_mean, RLw_noise_std = 0.0, 0.7
    RL_scale = 1
    reward_sigma, reward_window = 0.25, 25
    n_trials = reward_window + 10000                # Simulation parameters

    min_possible_output, max_possible_output = 0, 1                        # To calculate normalised error
    output_range = 1

    print('# --- INITIALISATIONS --- #')

    # Model structure #
    # --------------- #
    HVC = np.zeros(HVC_size)
    RA = np.zeros(RA_size)
    MO = np.zeros(MO_size)

#    W_HVC_RA_HL = np.zeros((HVC_size, RA_size), float) + Wmin_HL + (Wmax_HL-Wmin_HL)/2.0
    W_HVC_RA_HL = np.random.uniform(Wmin_HL + Wepsilon, Wmax_HL - Wepsilon, (HVC_size, RA_size))
    W_HVC_RA_RL = np.random.uniform(Wmin_RL + Wepsilon, Wmax_RL - Wepsilon, (HVC_size, RA_size))
    W_RA_MO = np.random.uniform(Wmin_MO + Wepsilon, Wmax_MO - Wepsilon, (RA_size, MO_size))

    # Model limits #
    # ------------ #

    # calculate output range
    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmin_HL

    RA[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    RA = sigmoid(RA, RA_sig_slope)

    MO[...] = np.dot(RA, W_RA_MO) / RA_size
    MO = sigmoid(MO, MO_sig_slope)

    min_possible_output = min(MO)

    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmax_HL


    RA[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    RA = sigmoid(RA, RA_sig_slope)

    MO[...] = np.dot(RA, W_RA_MO) / RA_size
    MO = sigmoid(MO, MO_sig_slope)

    max_possible_output = max(MO)

    output_range = max_possible_output - min_possible_output

    print("Output range:", output_range, min_possible_output, max_possible_output)


    # Syllable encoding and outputs #
    # ------------------------- #
    syllable_encoding = {}
    syllable_outputs = {}
#    syllables = ["A"]

    enc = np.zeros(HVC_size)
    enc[0] = 1
    syllable_encoding["A"] = enc

    outputs = np.random.uniform(min_possible_output + 0.01*output_range, max_possible_output - 0.01*output_range)
    syllable_outputs = {}
    syllable_outputs["A"] = outputs

    print('Syllable outputs:', syllable_outputs)

    # Simulation #
    # ---------- #
    E = np.zeros(n_trials)             # keeps track of error
    E_norm = np.zeros(n_trials)         # keeps track of normalised error
    R = np.zeros(n_trials)              # keeps track of reward
    Metric_R = 0.0                   # keeps track of final averaged reward

    # Learning song
    syll = "A"
    HVC[...] = syllable_encoding[syll]

    S_W_HR_HL = np.zeros((n_trials, W_HVC_RA_HL.size))               # to store HVC-RA weights
    S_W_HR_RL = np.zeros((n_trials, W_HVC_RA_RL.size))               # to store HVC-RA weights
    S_MO = np.zeros((n_trials, MO.size))

    # Learning syllable
    for nt in range(n_trials):
        if nt >= n_trials / 3.0:    Hebbian_learning = 1
        
        # Introducing noise for RL
        Noise = np.random.normal(RLw_noise_mean, RLw_noise_std, (HVC_size, RA_size))
        W_HVC_RA_RL_temp = W_HVC_RA_RL + Noise

        # Compute RA activity
        RA = np.zeros(RA_size)
        RA[...] += np.dot(HVC, W_HVC_RA_HL) / HVC_size * Hebbian_learning
        RA[...] += np.dot(HVC, W_HVC_RA_RL_temp) / HVC_size * Reinforcement_learning * RL_scale
        RA += np.random.normal(RA_noise_mean, RA_noise_std, RA_size)
        RA = sigmoid(RA, RA_sig_slope)

        # Compute MO activity
        MO[...] = np.dot(RA, W_RA_MO) / RA_size
        MO += np.random.normal(MO_noise_mean, MO_noise_std, MO_size)
        MO = sigmoid(MO, MO_sig_slope)

        # Compute error and reward
        error = np.sqrt(((MO - syllable_outputs[syll]) ** 2).sum()) / MO_size
        norm_error = error / output_range
        E[nt], E_norm[nt] = error, norm_error               # NOTE: Due to this a syllable can't repeat in a sequence
        R[nt] = np.exp(-norm_error ** 2 / reward_sigma ** 2)
        R_prev = 0

        # Compute update
        dW1 = pPos * HVC.reshape(HVC_size, 1) * (RA) * Hebbian_learning
        dW3 = pDec * (HVC.reshape(HVC_size, 1)) * (1-RA) * Hebbian_learning
        dW4 = dW1 * 0.0
        if nt > reward_window:
            R_prev = R[nt - reward_window: nt].sum() / float(reward_window)
            dW4 = eta * Noise * (R[nt] - R_prev) * HVC.reshape(HVC_size, 1) * (RA) * Reinforcement_learning * RL_scale

        W_HVC_RA_HL += dW1 * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
        W_HVC_RA_HL -= dW3 * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
        W_HVC_RA_RL += dW4 * (Wmax_RL - W_HVC_RA_RL) * (W_HVC_RA_RL - Wmin_RL)

        # Keeping track for plotting purposes
        S_W_HR_HL[nt] = W_HVC_RA_HL.ravel()
        S_W_HR_RL[nt] = W_HVC_RA_RL.ravel()
        S_MO[nt] = MO.ravel()


    print('Final MO for', syll, ':', MO, 'with target:', syllable_outputs[syll])
    print("Error:", E[n_trials-1], "Norm error:", E_norm[n_trials-1])
    print("Reward:", R[n_trials-1])
    Metric_R = R[nt - 500: nt].sum() / 500.0                       # For accuracy metrics
    print('Reward average over 500 learning trials:', Metric_R)



    # Plotting
    fig = plt.figure(figsize=(12, 8))

    T = np.arange(n_trials)

    ax = fig.add_subplot(4, 1, 1)

    ax.set_title("Testing RL")

    ax.plot(T, S_W_HR_HL)
    ax.set_xlabel('Trial no.')
    ax.set_ylabel('HVC-RA HL weights')
    ax.set_ylim(Wmin_HL, Wmax_HL)

    ax = fig.add_subplot(4, 1, 2)

    ax.plot(T, S_W_HR_RL)
    ax.set_xlabel('Trial no.')
    ax.set_ylabel('HVC-RA RL weights')
    ax.set_ylim(Wmin_RL, Wmax_RL)

    ax = fig.add_subplot(4, 1, 3)

    ax.plot(T, S_MO, label='Motor output', marker=',', markersize=1.0, linewidth=0, markerfacecolor='blue', alpha=.5)
    Target_OP = np.ones(n_trials) * syllable_outputs[syll]
    ax.set_xlabel('Trial no.')
    ax.set_ylabel('Averaged MO output')
    ax.set_ylim(min_possible_output, max_possible_output)

    # Plot a sliding average over 50 trials

    sw = reward_window * 2
    S_MO_sw = sliding_window(S_MO, sw)
    swT = range(len(S_MO_sw))
    ax.plot(swT, np.mean(S_MO_sw, axis=-1), linewidth=0.5, color="blue" )
    ax.plot(T, Target_OP, label = 'Target output', color="green")

    ax.legend(loc='lower right')

    ax = fig.add_subplot(4, 1, 4)

    ax.plot(T, E_norm, marker=',', markersize=1.0, linewidth=0, markerfacecolor='blue', alpha=.5)
    ax.plot(T, R, marker=',', markersize=1.0, linewidth=0, markerfacecolor='green', alpha=.5)
    ax.set_xlabel('Trial no.')
    ax.set_ylabel('Averaged Error/reward')
    ax.set_ylim(0, 1)

    # Plot a sliding average over 10 trials

    sw = 10
    E_norm_sw = sliding_window(E_norm, sw)
    R_sw = sliding_window(R, sw)
    swT = range(len(E_norm_sw))
    ax.plot(swT, np.mean(E_norm_sw, axis=-1), label = 'Norm Error', linewidth=0.5, color="blue")
    ax.plot(swT, np.mean(R_sw, axis=-1), label = 'Reward', linewidth=0.5, color="green")

    ax.legend(loc='lower right')

    plt.savefig(resFile + "_" + syll + ".png")
    # plt.show()
    plt.close()

    gc.collect()
    return Metric_R
