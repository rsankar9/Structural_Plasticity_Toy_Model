import single_syll
import numpy

f = 0
R = np.array([])
RT = np.array([])
Nruns = 5
for run in range(Nruns):
    f = f + 1
    rSeed = np.random.randint(0, 1e7)
    resFile = 'RResults/single_syll' + str(f)
    
    r_learn = single_syll.SPTModel(resFile, rSeed)
    R = np.append(R, r_learn)