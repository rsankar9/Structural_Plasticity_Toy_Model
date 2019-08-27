# HL with RL influence

import numpy as np
import matplotlib.pyplot as plt
import json
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
    arg_steepFactor = 5

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

    min_possible_output = 0                         # To calculate normalised error
    max_possible_output = 1
    output_range = 1

    n_testing_trials = 500                        # Testing parameters


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

    # calculate RA slope
    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmin_HL

    RA[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    min_RA_sig_in = np.min(RA)

    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmax_HL

    RA[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    max_RA_sig_in = np.max(RA)

    RA_sig_mid = (min_RA_sig_in + max_RA_sig_in) / 2.0
    RA_sig_slope = arg_steepFactor / (max_RA_sig_in - min_RA_sig_in)

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

#     Plotting sigmoids
    x_RA = np.linspace(min_RA_sig_in, max_RA_sig_in, 1000)
    x_MO = np.linspace(min_possible_output, max_possible_output, 1000)
    y_RA = sigmoid(x_RA, RA_sig_slope)
    y_MO = sigmoid(x_MO, MO_sig_slope)
    plt.plot(x_RA, y_RA, label='RA')
    plt.plot(x_MO, y_MO, label='MO')
    plt.legend()
    plt.show()


    # Syllable encoding and outputs #
    # ------------------------- #
    syllable_encoding = {}
    syllable_outputs = {}
    syllables = ["A", "B", "C"]
    seq = ["A", "B", "C"]               # Note: No repeating syllables allowed


    for i in range(len(syllables)):
        enc = np.zeros(HVC_size)
        enc[i] = 1
        syllable_encoding[syllables[i]] = enc

    outputs = np.random.uniform(min_possible_output + 0.01*output_range, max_possible_output - 0.01*output_range, (len(syllables), MO_size))
    syllable_outputs = {}
    for i in range(len(syllables)):
        syllable_outputs[syllables[i]] = outputs[i]

    print('Syllable outputs:', syllable_outputs)

    # Simulation #
    # ---------- #
    E = np.zeros((len(seq), n_trials))              # keeps track of error
    E_norm = np.zeros((len(seq), n_trials))         # keeps track of normalised error
    E_norm_clear = np.zeros((len(seq), n_trials))   # keeps track of error without noisy weights
    R = np.zeros((len(seq), n_trials))              # keeps track of reward
    Metric_R = np.zeros(len(seq))                   # keeps track of final averaged reward

    # Learning song
    for seq_no in range(len(seq)):
        syll = seq[seq_no]
        print('Learning syll:', syll, 'Target output:', syllable_outputs[syll])

        HVC[...] = syllable_encoding[syll]
        # W_HVC_RA_HL = np.zeros((HVC_size, RA_size), float) + Wepsilon
        # W_HVC_RA_RL = np.random.uniform(Wmin_RL + Wepsilon, Wmax_RL - Wepsilon, (HVC_size, RA_size))

        S_W_HR_HL = np.zeros((n_trials, W_HVC_RA_HL.size))               # to store HVC-RA weights
        S_W_HR_RL = np.zeros((n_trials, W_HVC_RA_RL.size))               # to store HVC-RA weights
        S_MO = np.zeros((n_trials, MO.size))
        S_MO_clear = np.zeros((n_trials, MO.size))

        # Learning syllable
        for nt in range(n_trials):

            if nt > n_trials / 3.0: Hebbian_learning = 1

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
            E[seq_no, nt], E_norm[seq_no, nt] = error, norm_error               # NOTE: Due to this a syllable can't repeat in a sequence
            R[seq_no, nt] = np.exp(-norm_error ** 2 / reward_sigma ** 2)
            R_prev = 0

            # Computing error without noisy weights
            RA_clear = np.zeros(RA_size)
            RA_clear[...] += np.dot(HVC, W_HVC_RA_HL) / HVC_size * Hebbian_learning
            RA_clear[...] += np.dot(HVC, W_HVC_RA_RL) / HVC_size * Reinforcement_learning * RL_scale
            RA_clear += np.random.normal(RA_noise_mean, RA_noise_std, RA_size)
            RA_clear = sigmoid(RA_clear, RA_sig_slope)
            MO_clear = np.zeros(MO_size)
            MO_clear[...] = np.dot(RA_clear, W_RA_MO) / RA_size
            MO_clear[...] += np.random.normal(MO_noise_mean, MO_noise_std, MO_size)
            MO_clear[...] = sigmoid(MO_clear, MO_sig_slope)
#            if nt % 100 == 0: print('MO clear', MO_clear)
            error_clear = np.sqrt(((MO_clear - syllable_outputs[syll]) ** 2).sum()) / MO_size
            norm_error_clear = error_clear / output_range
            E_norm_clear[seq_no, nt] = norm_error_clear

            # Compute update
            # dW1 = pPos * HVC.reshape(HVC_size, 1) * (RA>=np.median(RA)) * Hebbian_learning
            # dW2 = pDec * (1 - HVC.reshape(HVC_size, 1)) * (RA>=np.median(RA)) * Hebbian_learning
            # dW3 = pDec * (HVC.reshape(HVC_size, 1)) * (RA<np.median(RA)) * Hebbian_learning
            # dW4 = dW1 * 0.0
            # if nt > reward_window:
            #     R_prev = R[seq_no][nt - reward_window: nt].sum() / float(reward_window)
            #     dW4 = eta * Noise * (R[seq_no, nt] - R_prev) * HVC.reshape(HVC_size,1) * (RA) * Reinforcement_learning

            # Compute update
            dW1 = pPos * HVC.reshape(HVC_size, 1) * (RA) * Hebbian_learning
#            dW2 = pDec * (1 - HVC.reshape(HVC_size, 1)) * (RA) * Hebbian_learning
            dW3 = pDec * (HVC.reshape(HVC_size, 1)) * (1-RA) * Hebbian_learning
            dW4 = dW1 * 0.0
            if nt > reward_window:
                R_prev = R[seq_no][nt - reward_window: nt].sum() / float(reward_window)
                dW4 = eta * Noise * (R[seq_no, nt] - R_prev) * HVC.reshape(HVC_size, 1) * (RA) * Reinforcement_learning * RL_scale
#            if nt%100 == 0: print('Noise:', Noise, 'RA:', RA)
#            if nt%100 == 0: print('dW1:', dW1, 'dW2:', dW2, 'dW3:', dW3, 'dW4:', dW4)

            W_HVC_RA_HL += dW1 * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
#            W_HVC_RA_HL -= dW2 * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
            W_HVC_RA_HL -= dW3 * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
            W_HVC_RA_RL += dW4 * (Wmax_RL - W_HVC_RA_RL) * (W_HVC_RA_RL - Wmin_RL)

            # Keeping track for plotting purposes
            S_W_HR_HL[nt] = W_HVC_RA_HL.ravel()
            S_W_HR_RL[nt] = W_HVC_RA_RL.ravel()
            S_MO[nt] = MO.ravel()
            S_MO_clear[nt] = MO_clear.ravel()


        print('Final MO for', syll, ':', MO, 'with target:', syllable_outputs[syll])
        print("Error:", E[seq_no, n_trials-1], "Norm error:", E_norm[seq_no, n_trials-1])
        print("Reward:", R[seq_no, n_trials-1])
        Metric_R[seq_no] = R[seq_no][nt - 500: nt].sum() / 500.0                       # For accuracy metrics
        print('Reward average over 500 learning trials:', Metric_R[seq_no])



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
        ax.plot(T, S_MO_clear, label='Clear motor output', marker=',', markersize=1.0, linewidth=0, markerfacecolor='orange', alpha=.5)
        Target_OP = np.ones(n_trials) * syllable_outputs[syll]
        ax.set_xlabel('Trial no.')
        ax.set_ylabel('Averaged MO output')
        ax.set_ylim(min_possible_output, max_possible_output)

        # Plot a sliding average over 50 trials

        sw = reward_window * 2
        S_MO_sw = sliding_window(S_MO, sw)
        S_MO_clear_sw = sliding_window(S_MO_clear, sw)
        swT = range(len(S_MO_sw))
        ax.plot(swT, np.mean(S_MO_sw, axis=-1), linewidth=0.5, color="blue" )
        ax.plot(swT, np.mean(S_MO_clear_sw, axis=-1), linewidth=0.5, color="orange")
        ax.plot(T, Target_OP, label = 'Target output', color="green")

        ax.legend(loc='lower right')

        ax = fig.add_subplot(4, 1, 4)

        ax.plot(T, E_norm[seq_no], marker=',', markersize=1.0, linewidth=0, markerfacecolor='blue', alpha=.5)
        ax.plot(T, E_norm_clear[seq_no], marker=',', markersize=1.0, linewidth=0, markerfacecolor='orange', alpha=.5)
        ax.plot(T, R[seq_no], marker=',', markersize=1.0, linewidth=0, markerfacecolor='green', alpha=.5)
        ax.set_xlabel('Trial no.')
        ax.set_ylabel('Averaged Error/reward')
        ax.set_ylim(0, 1)

        # Plot a sliding average over 10 trials

        sw = 10
        E_norm_sw = sliding_window(E_norm[seq_no], sw)
        E_norm_clear_sw = sliding_window(E_norm_clear[seq_no], sw)
        R_sw = sliding_window(R[seq_no], sw)
        swT = range(len(E_norm_sw))
        ax.plot(swT, np.mean(E_norm_sw, axis=-1), label = 'Norm Error', linewidth=0.5, color="blue")
        ax.plot(swT, np.mean(E_norm_clear_sw, axis=-1), label = 'Clear Norm Error', linewidth=0.5, color="orange")
        ax.plot(swT, np.mean(R_sw, axis=-1), label = 'Reward', linewidth=0.5, color="green")

        ax.legend(loc='lower right')

        plt.savefig(resFile + "_" + syll + ".png")
        # plt.show()
        plt.close()

    print("------ Testing ------")
    R_test_avg = np.zeros(len(seq))
    seq_output = np.array([])
    for seq_no in range(len(seq)):
        syll = seq[seq_no]
        print('Testing syll:', syll, 'Target output:', syllable_outputs[syll])

        HVC[...] = syllable_encoding[syll]
        # Testing syllable
        R_test_sum = 0.0
        MO_test = np.array([])
        for ntt in range(n_testing_trials):

            # Compute RA activity
            RA = np.zeros(RA_size)
            RA[...] += np.dot(HVC, W_HVC_RA_HL) / HVC_size * Hebbian_learning
            # RA[...] += np.dot(HVC, W_HVC_RA_RL_temp) / HVC_size * Reinforcement_learning
            RA += np.random.normal(RA_noise_mean, RA_noise_std, RA_size)
            RA = sigmoid(RA, RA_sig_slope)

            # Compute MO activity
            MO[...] = np.dot(RA, W_RA_MO) / RA_size
            MO += np.random.normal(MO_noise_mean, MO_noise_std, MO_size)
            MO = sigmoid(MO, MO_sig_slope)

            # Compute error and reward
            error = np.sqrt(((MO - syllable_outputs[syll]) ** 2).sum()) / MO_size
            norm_error = error / output_range
            R_test = np.exp(-norm_error ** 2 / reward_sigma ** 2)
            R_test_sum += R_test

            MO_test = np.append(MO_test, MO)

        avg_MO = np.mean(MO_test)

        min_err = 1000.0
        min_err_syll = 'Q'

        for s in seq:
            err_curr = (np.sqrt(((MO - syllable_outputs[s]) ** 2).sum()) / MO_size)
            if err_curr < min_err:
                min_err, min_err_syll = err_curr, s
        seq_output = np.append(seq_output, min_err_syll)
        print('Mean MO:', avg_MO)
        print('Syllable outputs:', syllable_outputs)

        R_test_avg[seq_no] = R_test_sum/float(n_testing_trials)
        print('Reward average over 500 test trials:', R_test_avg[seq_no])


    # --- Write to JSON file --- #

    layer_parameters = {
        "HVC population": HVC_size,
        "RA population": RA_size,
        "MC population": MO_size,
    }

    activation_function_parameters = {
        "RA noise mean": RA_noise_mean,
        "RA noise sigma": RA_noise_std,
        "MO noise mean": MO_noise_mean,
        "MO noise sigma": MO_noise_std,
    }

    learning_parameters = {
        "No. of trials": n_trials,
        "No. of training trials": n_testing_trials,
        "Hebbian learning (ON/OFF)": Hebbian_learning,
        "Hebbian learning rate": pPos,
        "Hebbian decay rate": pDec,
        "Reinforcement learning (ON/OFF)": Reinforcement_learning,
        "RL learning rate": eta,
        "RL weights noise mean": RLw_noise_mean,
        "RL weights noise std": RLw_noise_std,
        "Reward sigma": reward_sigma,
        "Reward window": reward_window
    }

    bound_parameters = {
        "Hebbian weight lower": Wmin_HL,
        "Hebbian weight upper": Wmax_HL,
        "RL weight lower": Wmin_RL,
        "RL weight upper": Wmax_RL,
        "MO weight lower (init)": Wmin_MO,
        "MO weight upper (init)": Wmax_MO,
        "Weight margin (init)": Wepsilon,
        "Output lower": min_possible_output,
        "Output upper": max_possible_output,
        "Output range": output_range,
        "RA sigmoid slope": RA_sig_slope,
        "MC sigmoid slope": MO_sig_slope,
        "RA sigmoid steepness factor": arg_steepFactor
    }

    sequences_parameters = {
        # "Training Sequences": training_sequences,
        "Target output": str(syllable_outputs)
    }

    input_parameters = {
        "Layer Parameters": layer_parameters,
        "Activation Function Parameters": activation_function_parameters,
        "Learning Parameters": learning_parameters,
        "Bounds": bound_parameters,
        "Sequences": sequences_parameters
    }

    results = {
        "Output range": str([output_range, min_possible_output, max_possible_output]),
        "Final error": str([E[:,n_trials - 1]]),
        "Final normalised error": str([E_norm[:,n_trials - 1]]),
        "Final reward": str([R[:, n_trials - 1]]),
        "Final mean reward over n learning trials": str(Metric_R),
        "Final mean reward over n testing trials": str(R_test_avg)
    }

    Data = {
        "Purpose": "Learning Syllables with RL only",
        # "GitHash": "",
        "Random seed": rSeed,
        "Input": input_parameters,
        "Results": results,
        "Remark": "First try"
    }

    print("Writing to json:", resFile)
    with open(resFile + ".json", 'w') as outfile:
        json.dump(Data, outfile, sort_keys=False, indent=4, separators=(',', ':\t'))

    gc.collect()
    return Metric_R, R_test_avg, seq, seq_output
