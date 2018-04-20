# Untangle Mersenne
Trying to predict the seed of a pseudorandom sequence of numbers with a simple perceptron.
# Background
The Mersenne Twister is a pseudorandom number generator (PRNG). It is by far the most widely used general-purpose PRNG. Its name derives from the fact that its period length is chosen to be a Mersenne prime.
# Method
Generate lists of samples using multiple seeds. Label those according to the seed and train a multilayer perceptron on the task. Generate new samples. Test the model. Regret.

![Graph](https://github.com/paubric/python-untangle-mersenne/blob/master/Nope2.png)

Nope.

# TO DO
- Wonder why
