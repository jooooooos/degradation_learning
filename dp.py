import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from tqdm.notebook import tqdm, trange
from numba import njit, prange

np.set_printoptions(suppress=True)

N = 100
precision = 2

max_degradation = 50
holding_cost = 5 # per-timeunit holding cost
replacement_cost = 11 # replacement const
failure_cost = 5 # failure cost

contexts = np.arange(0, 10, 1 / N).round(precision)
states = np.arange(0, max_degradation, 1/N).round(precision)

RENT, SHUTDOWN, REPLACE = 0, 1, 2

actions = [
    RENT,
    SHUTDOWN,
    REPLACE
]

n_states = len(states)
n_contexts = len(contexts)
n_actions = len(actions)

P = np.zeros((n_states, n_contexts, n_actions, n_states), dtype=np.float32)
R = np.zeros((n_states, n_contexts, n_actions, n_states), dtype=np.float32)

d_inds = np.arange(1, max_k + 1)

# initialize the transition and reward matrices

for s_ind, s in tqdm(enumerate(states)):
    for a_ind, a in enumerate(actions):
        if a == SHUTDOWN:
            if s_ind == n_states - 1:
                # machine breaks down on its own
                P[s_ind, :, a_ind, 0] = 1
                R[s_ind, :, a_ind, 0] = -replacement_cost - holding_cost
                
                # # # machine can't fail but still incurs holding cost
                # P[s_ind, :, a_ind, s_ind] = 1
                # R[s_ind, :, a_ind, s_ind] = -holding_cost
            
            elif s_ind + max_k < n_states:
                # machine undergoes time-based degradation, 
                # and can fail due to time-based degradation.
                
                failure_probs = failure_probability(s, tbd_x)
                failure_prob = (tbd_pmf * failure_probs).sum()

                # surviving time-based degradation
                P[s_ind, :, a_ind, s_ind+d_inds] = np.tile(
                    tbd_pmf * (1- failure_probs),
                    (n_contexts, 1)
                ).T
                R[s_ind, :, a_ind, s_ind+d_inds] = -holding_cost
                
                # failing due to time-based degradation
                P[s_ind, :, a_ind, 0] = failure_prob
                R[s_ind, :, a_ind, 0] = -replacement_cost - holding_cost
                
            else:
                # time-degradataion can reach the maximum degradation level
                # probabilities of transitions that lead to the maximum degradation level or above
                # are summed up and assigned to probability of transition to the maximum degradation level
                
                split_index = np.where(s_ind + d_inds == n_states-1)[0][0]
                tbd_pmf_before = tbd_pmf[:split_index]
                tbd_pmf_after_sum = tbd_pmf[split_index:].sum()
                tbd_pmf_new = np.append(tbd_pmf_before, tbd_pmf_after_sum)

                failure_probs = failure_probability(s, tbd_x[:len(tbd_pmf_new)])
                failure_prob = (tbd_pmf_new * failure_probs).sum()

                # surviving time-based degradation
                P[s_ind, :, a_ind, -len(tbd_pmf_new):] = np.tile(
                    tbd_pmf_new * (1- failure_probs),
                    (n_contexts, 1)
                )
                R[s_ind, :, a_ind, -len(tbd_pmf_new):] = -holding_cost
                
                # failing due to time-based degradation
                P[s_ind, :, a_ind, 0] = failure_prob
                R[s_ind, :, a_ind, 0] = -replacement_cost - holding_cost

        elif a == REPLACE:
            # machine is replaced and the degradation level is reset to 0
            P[s_ind, :, a_ind, 0] = 1
            R[s_ind, :, a_ind, 0] = -replacement_cost
            
        elif a == RENT:
            for x_ind, x in enumerate(contexts):
                ns = round(s + x, precision)
                ind_ns = round(ns * 100) if ns <= states[-1] else -1
                
                # machine is rented out and fails
                P[s_ind, x_ind, a_ind, 0] = failure_probability(s, x)
                R[s_ind, x_ind, a_ind, 0] = u * x - replacement_cost - failure_cost
                
                # machine is rented out and survives
                P[s_ind, x_ind, a_ind, ind_ns] = 1 - P[s_ind, x_ind, a_ind, 0]
                R[s_ind, x_ind, a_ind, ind_ns] = u * x                
                
def value_iteration(P, R, V=None, policy=None, gamma=1.0, theta=1e-8, max_iter=10000):
    """
    Perform value iteration for an MDP to find the optimal policy and value function.
    - N+1 states (0..N), where N is terminal
    - M contexts (0..M-1)
    - 2 actions (0, 1)

    Parameters:
        P (numpy.ndarray): Transition probabilities
            4D array of shape (n_states, n_contexts, n_actions, n_states)
        R (numpy.ndarray): Rewards
            4D array of shape (n_states, n_contexts, n_actions, n_states)
        gamma (float): Discount factor
        theta (float): Threshold for convergence
        max_iter (int): Maximum number of iterations
        V (numpy.ndarray): Initial value function (optional)
            If None, it will be initialized to zeros.
        policy (numpy.ndarray): Initial policy (optional)
            If None, it will be initialized to zeros.

    Returns:
        V (numpy.ndarray): Optimal value function
        policy (numpy.ndarray): Optimal policy
    """
    n_states, n_contexts, n_actions = P.shape[0], P.shape[1], P.shape[2]
    
    # V and policy must be provided together or not at all
    if V is not None and policy is not None:
        assert V.shape == (n_states, n_contexts), "V must have shape (n_states, n_contexts)"
        assert policy.shape == (n_states, n_contexts), "policy must have shape (n_states, n_contexts)"
    elif V is not None:
        assert V.shape == (n_states, n_contexts), "V must have shape (n_states, n_contexts)"
        policy = np.zeros((n_states, n_contexts), dtype=int)
    elif policy is not None:
        assert policy.shape == (n_states, n_contexts), "policy must have shape (n_states, n_contexts)"
        V = np.zeros((n_states, n_contexts))
    else:
        # If neither V nor policy is provided, initialize both
        V = np.zeros((n_states, n_contexts))
        policy = np.zeros((n_states, n_contexts), dtype=int)

    for _ in trange(max_iter):
        delta = 0.0
        V_prev = V.copy()
        
        for s in range(n_states):
            for x in range(n_contexts):
                # compute the q-value for each action
                # Q(a) = sum_{s_next} P[s,x,a,s_next] * (R[s,x,a,s_next] + gamma * V_prev[s_next,x])
                
                Q_values = np.zeros(n_actions)
                for a in range(n_actions):
                    expected_reward = np.sum(P[s, x, a, :] * R[s, x, a, :])
                    expected_future = P[s, x, a, :] @ V_prev.mean(avis=1)
                    Q_values[a] = expected_reward + gamma * expected_future
                    
                # Best action and best value
                best_action = np.argmax(Q_values)
                best_value = Q_values[best_action]
                V[s, x] = best_value
                policy[s, x] = best_action
                delta = max(delta, abs(V[s, x] - V_prev[s, x]))
        # Check for convergence
        if delta < theta:
            break
        
    return V, policy

# V, policy = value_iteration(P, R, V=V, policy=policy, gamma=1.0, theta=1e-6, max_iter=1000)

@njit(parallel=True)
def value_iteration_numba(P, R, gamma, theta, max_iter):
    n_states, n_contexts, n_actions, _ = P.shape
    V = np.zeros((n_states, n_contexts))
    policy = np.zeros((n_states, n_contexts), dtype=np.int64)

    for _ in range(max_iter):
        delta = 0.0
        V_prev = V.copy()

        V_prev_mean = np.zeros(n_states)
        for s in range(n_states):
            for x in range(n_contexts):
                V_prev_mean[s] += V_prev[s, x]
            V_prev_mean[s] /= n_contexts
        
        # We can parallelize over s and x, but must do a 2D loop
        for s in prange(n_states):
            for x in range(n_contexts):
                # compute Q-values
                Q_values = np.zeros(n_actions)
                for a_ind in range(n_actions):
                    reward = 0
                    future = 0
                    for s_next in range(n_states):
                        reward += P[s, x, a_ind, s_next] * R[s, x, a_ind, s_next]
                        future += P[s, x, a_ind, s_next] * V_prev_mean[s_next]

                    Q_values[a_ind] = reward + gamma * future                        
                        
                best_a = np.argmax(Q_values)
                best_v = Q_values[best_a]
                policy[s, x] = best_a
                V[s, x] = best_v
                diff = abs(best_v - V_prev[s,x])
                
                if diff > delta:
                    delta = diff
        if delta < theta:
            break

    return V, policy

V, policy = value_iteration_numba(
    P, R,
    gamma=1, theta=1e-8, max_iter=10000
)