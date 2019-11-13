#!/usr/bin/env python
import glob
import pickle
import random
import os.path as op

import numpy as np


# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000


def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter_ = int(op.splitext(op.basename(f))[0].split("_")[2])
        if iter_ > st:
            st = iter_

    if st > 0:
        params_file = "saved_params_{}.npy".format(st)
        state_file = "saved_state_{}.pickle".format(st)
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter_, params):
    params_file = "saved_params_{}.npy".format(iter_)
    np.save(params_file, params)
    with open("saved_state_{}.pickle".format(iter_), "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False,
        print_every=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    print_every -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """
    # Anneal learning rate every several iterations
    anneal_every = 20000

    if use_saved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / anneal_every)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    for iter_ in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        loss = None
        ### YOUR CODE HERE
        raise NotImplementedError
        ### END YOUR CODE

        x = postprocessing(x)
        if iter_ % print_every == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter_, exploss))

        if iter_ % SAVE_PARAMS_EVERY == 0 and use_saved:
            save_params(iter_, x)

        if iter_ % anneal_every == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, print_every=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, print_every=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, print_every=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()
