## Review Illustration

Taking inspiration from the songbird literature, we see that a primary cortical pathway is responsible for producing song. However, learning a song, similar to the tutor song, is assisted by a secondary BG-thalamocortical pathway. Meanwhile, the primary pathway sprouts new synaptic connections.

We illustrate this concept here, in simpler terms.

Instead of a song, we use a combination of 2 sine waves, of different frequencies, as a target.
We use simplified, unrealistic means to represent the 2 pathways.

The system has 2 pathways:-
- Pathway 1: Primary cortical pathway, producing song.
- Pathway 2: Secondary BG-thalamocortical pathway, providing a tutor signal.

Pathway 2 processes the output of the system, and evaluates the reward obtained. Here, we use reinforcement learning.

Pathway 1 uses the output of pathway 2 as a tutor signal, and tries to mimic it.

Pathway 1 and 2 are responsible for accurately reaching the desired amplitude of the sine wave of the faster frequency.

Meanwhile, independently, the amplitude of the slower sine wave grows automatically (not modelled here).

In the results, you can see that:-
- Pathway 2 initially explores, and after having coached pathway 1, it's activity reduces to zero.
- Pathway 1 follows pathway 2, and eventually produces the correct output.
- However, this happens more robustly, when there's a delay in the development of pathway 1.
- When pathway 1 and 2 are updated simulataneously, from the very beginning, it tends to disrupt the learning, and sometimes never converges.


### Results

1. Slow development; Negative values allowed
![Result 1a](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_slow_neg/Result_illustration.png)
![Result 1b](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_slow_neg/Result_illustration_delayed.png)

2. Fast development; Negative values allowed
![Result 2a](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_fast_neg/Result_illustration.png)
![Result 2b](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_fast_neg/Result_illustration_delayed.png)

3. Slow development; Negative values not allowed
![Result 3a](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_slow_noneg/Result_illustration.png)
![Result 3b](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_slow_noneg/Result_illustration_delayed.png)

4. Fast development; Negative values not allowed
![Result 4a](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_fast_noneg/Result_illustration.png)
![Result 4b](/Users/rsankar/Documents/Work/Ongoing work/Toy Illustration/Trial3/Works_fast_noneg/Result_illustration_delayed.png)