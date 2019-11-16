import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x = x - np.max(x, axis=1)[:, None]
        x = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x = x-np.max(x)
        x = np.exp(x)/np.sum(np.exp(x))
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x

def some_test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    test1 = softmax(np.array([0, 0, 0, 0]))
    print(test1)
    ans1 = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[0, 0, 0, 0], [3, 3, 3, 3], [5, 5, 5, 5]]))
    print(test2)
    ans2 = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)
    ### END YOUR CODE


if __name__ == "__main__":
    some_test_softmax_basic()
    your_softmax_test()
