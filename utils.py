from sklearn.utils import Bunch
import numpy as np
import math

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)

def make_spirals(n_samples=150, noise=0.1, random_state=None):
    """
    Generate a dataset of spirals.

    Arguments:
        n_samples: (int, default=150) The total number of points equally divided among two spirals.
        noise: (float, default=0.1) The standard deviation of the Gaussian noise added to the data.
        random_state: (int, RandomState instance, or None, default=None) Controls the randomness of the dataset.
                      Pass an int for reproducible output across multiple function calls.
                      Pass a RandomState instance to use it as random number generator.
                      Pass None (default) to use the global numpy random number generator.

    Returns:
        A sklearn Bunch object containing the generated dataset.
    """
    rng = np.random.default_rng(random_state)

    X = np.empty((n_samples, 2), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int32)

    for i in range(n_samples // 2):
        X[i], X[i + n_samples // 2] = [spiral_xy(i, 1), spiral_xy(i, -1)]
        y[i], y[i + n_samples // 2] = 0, 1

    X += rng.normal(scale=noise, size=X.shape)
    X -= X.min(axis=0)
    X /= X.max(axis=0)

    feature_names = ['x', 'y']
    target_names = ['Class 0', 'Class 1']

    return Bunch(data=X,
                 target=y,
                 feature_names=feature_names,
                 target_names=target_names)
