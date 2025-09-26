import numpy as np

def unit_ball_sample(d):
    while True:
        x = np.random.randn(d)
        if np.linalg.norm(x) <= 1:
            return x

def correct_signs(u, c, d):
    """
    Args:
        u: (d,) array of agent utilities
        c: (d,) array of context
        d: int dimension of the context

    Returns:
        Flip signs of c until u @ c > 0
    """

    while u @ c < 0:
        c = -c
    return c