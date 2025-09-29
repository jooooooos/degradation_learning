import numpy as np
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import logging

def delta_bar(T, d):
    temp = 16 * T ** 2 * d * (d+1) ** 2
    return 1 / temp

def _solve_s_optimization(S, x, d, sense):
    """
    Helper function to solve for max or min of <s, x> over the set S.
    This is the core optimization problem.

    Args:
        S (dict): The set defined by {"w": list_of_np_arrays, "x": list_of_floats, "sign": list_of_str ('ge' or 'le')}.
        x (np.array): The direction vector for the objective.
        d (int): The dimension of the space.
        sense (GRB.MAXIMIZE or GRB.MINIMIZE): The optimization sense.

    Returns:
        float: The optimal objective value, or None if failed.
    """
    try:
        m = gp.Model("s_optimizer")
        s = m.addMVar(shape=d, lb=-1.0, ub=1.0, name="s")
        m.setObjective(x @ s, sense)
        m.addConstr(s @ s <= 1.0, name="norm_constraint")
        for j, (w_j, x_j, sign_j) in enumerate(zip(S["w"], S["x"], S["sign"])):
            if sign_j == "ge":
                m.addConstr(w_j @ s >= x_j, name=f"half_space_{j}")
            elif sign_j == "le":
                m.addConstr(w_j @ s <= x_j, name=f"half_space_{j}")
            else:
                raise ValueError(f"Invalid sign: {sign_j}")
        m.setParam('OutputFlag', 0)
        m.optimize()
        if m.status == GRB.OPTIMAL:
            return m.ObjVal
        else:
            logging.info(f"Optimization failed with status: {m.status}")
            return None
    except gp.GurobiError as e:
        logging.info(f"Error code {e.errno}: {e}")
        return None

def diam(S, x, d):
    """
    Computes the diameter of the set S in the direction x.
    diam(S_t, x) = max_{s1, s2 in S_t} |<s1 - s2, x>| for any x in R^d.

    Args:
        S (dict): The set defined by {"w": list_of_np_arrays, "x": list_of_floats, "sign": list_of_str ('ge' or 'le')}.
        x (np.array): The direction vector.
        d (int): The dimension of the space.

    Returns:
        float: The computed diameter, or None if failed.
    """
    max_val = _solve_s_optimization(S, x, d, GRB.MAXIMIZE)
    min_val = _solve_s_optimization(S, x, d, GRB.MINIMIZE)
    if max_val is not None and min_val is not None:
        return max_val - min_val
    else:
        return None
    
def find_point_in_S(S, d):
    """
    Finds a feasible point in S using Gurobi (for initial point in hit-and-run).

    Args:
        S (dict): The set S.
        d (int): Dimension.

    Returns:
        np.array: A point in S, or None if S is empty.
    """
    try:
        m = gp.Model("feasibility")
        s = m.addMVar(shape=d, lb=-1.0, ub=1.0, name="s")
        m.setObjective(0, GRB.MINIMIZE)  # Dummy objective
        m.addConstr(s @ s <= 1.0, name="norm_constraint")
        for j, (w_j, x_j, sign_j) in enumerate(zip(S["w"], S["x"], S["sign"])):
            if sign_j == "ge":
                m.addConstr(w_j @ s >= x_j, name=f"half_space_{j}")
            elif sign_j == "le":
                m.addConstr(w_j @ s <= x_j, name=f"half_space_{j}")
        m.setParam('OutputFlag', 0)
        m.optimize()
        if m.status == GRB.OPTIMAL:
            return np.array(s.X)
        else:
            return None
    except gp.GurobiError:
        return None

def feasibility_check(y, V, mins, maxs, half_spaces, tol=1e-6):
    """
    Checks if point y is in Cyl(S, V).

    Args:
        y (np.array): Point to check.
        V (list of np.array): Orthonormal vectors.
        mins (list): Min projections along each v_i.
        maxs (list): Max projections along each v_i.
        half_spaces (list of tuples): (w, x, sign) for constraints.
        tol (float): Numerical tolerance.

    Returns:
        bool: True if y in Cyl(S, V).
    """
    n = len(V)
    # Check interval projections
    for i, v in enumerate(V):
        proj = np.dot(y, v)
        if proj < mins[i] - tol or proj > maxs[i] + tol:
            return False
    # Compute z = projection onto V_perp
    z = y.copy()
    for v in V:
        z -= np.dot(y, v) * v
    r_sq = 1 - np.linalg.norm(z)**2
    if r_sq < -tol:
        return False
    r = np.sqrt(max(r_sq, 0))
    if n == 0:
        # Special case: no V, check directly on y
        if np.linalg.norm(y) > 1 + tol:
            return False
        for w, x, sign in half_spaces:
            proj = np.dot(y, w)
            if sign == 'ge' and proj < x - tol:
                return False
            if sign == 'le' and proj > x + tol:
                return False
        return True
    # Set up constraints for beta in R^n
    constraints = [{'type': 'ineq', 'fun': lambda beta: r - np.linalg.norm(beta) + tol}]
    for w, x, sign in half_spaces:
        a = np.array([np.dot(v, w) for v in V])
        const = np.dot(z, w)
        if sign == 'ge':
            constraints.append({'type': 'ineq', 'fun': lambda beta, const=const, a=a, x=x: const + np.dot(beta, a) - x + tol})
        elif sign == 'le':
            constraints.append({'type': 'ineq', 'fun': lambda beta, const=const, a=a, x=x: x - (const + np.dot(beta, a)) + tol})
    # Feasibility: minimize 0 subject to constraints
    res = minimize(fun=lambda beta: 0, x0=np.zeros(n), constraints=constraints, method='SLSQP', tol=tol)
    return res.success

def find_min_lambda(p, dir, feasibility_func, tol=1e-6, max_bound=1e4):
    """
    Finds the minimal lambda for the line p + lambda * dir in the set (left boundary).

    Args:
        p (np.array): Current point.
        dir (np.array): Unit direction.
        feasibility_func (callable): Feasibility check.
        tol (float): Tolerance.
        max_bound (float): Max search bound to prevent infinite loop.

    Returns:
        float: Min lambda.
    """
    lambda_cur = 0.0
    step = 0.1
    feasible_lambda = 0.0
    while abs(lambda_cur) < max_bound:
        if not feasibility_func(p + lambda_cur * dir):
            break
        feasible_lambda = lambda_cur
        step *= 2
        lambda_cur -= step
    if abs(lambda_cur) >= max_bound:
        raise ValueError("Set appears unbounded in negative direction")
    low = lambda_cur  # infeasible
    high = feasible_lambda  # feasible
    while abs(high - low) > tol:
        mid = (low + high) / 2
        if feasibility_func(p + mid * dir):
            high = mid  # push left
        else:
            low = mid
    return high

def find_max_lambda(p, dir, feasibility_func, tol=1e-6, max_bound=1e4):
    """
    Finds the maximal lambda for the line p + lambda * dir in the set (right boundary).

    Args:
        Same as find_min_lambda.

    Returns:
        float: Max lambda.
    """
    lambda_cur = 0.0
    step = 0.1
    feasible_lambda = 0.0
    while abs(lambda_cur) < max_bound:
        if not feasibility_func(p + lambda_cur * dir):
            break
        feasible_lambda = lambda_cur
        step *= 2
        lambda_cur += step
    if abs(lambda_cur) >= max_bound:
        raise ValueError("Set appears unbounded in positive direction")
    low = feasible_lambda  # feasible
    high = lambda_cur  # infeasible
    while abs(high - low) > tol:
        mid = (low + high) / 2
        if feasibility_func(p + mid * dir):
            low = mid  # push right
        else:
            high = mid
    return low

def hit_and_run(num_samples, thin, burn_in, d, initial_p, feasibility_func, tol=1e-8):
    """
    Hit-and-run sampler to generate approximate uniform samples from the convex set.

    Args:
        num_samples (int): Number of samples to collect.
        thin (int): Thinning interval (samples to skip between collections).
        burn_in (int): Number of burn-in steps.
        d (int): Dimension.
        initial_p (np.array): Initial feasible point.
        feasibility_func (callable): Feasibility check function.
        tol (float): Tolerance for boundary search.

    Returns:
        np.array: Array of samples (num_samples x d).
    """
    samples = []
    p = initial_p.copy()
    total_steps = burn_in + num_samples * thin
    for step in range(total_steps):
        dir = np.random.randn(d)
        dir /= np.linalg.norm(dir) + 1e-10  # Avoid division by zero
        min_l = find_min_lambda(p, dir, feasibility_func, tol)
        max_l = find_max_lambda(p, dir, feasibility_func, tol)
        if min_l >= max_l - tol:
            logging.warning("Degenerate step in hit-and-run; skipping sample.")
            continue
        lambda_new = np.random.uniform(min_l, max_l)
        p = p + lambda_new * dir
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(p.copy())
    return np.array(samples), p

def get_centroid(S, V, d, num_samples=2000, thin=None, burn_in=None, tol=1e-6, rho_target=0.01, return_samples=False):
    """
    Computes an approximate centroid of Cyl(S, V) using hit-and-run sampling.

    Args:
        S (dict): Set {"w": list_of_np_arrays, "x": list_of_floats, "sign": list_of_str ('ge' or 'le')}.
        V (list of np.array): Orthonormal vectors.
        d (int): Dimension.
        num_samples (int): Number of samples for averaging.
        thin (int, optional): Thinning; defaults to d.
        burn_in (int, optional): Burn-in steps; defaults to 100 * d**2.
        tol (float): Numerical tolerance.
        rho_target (float): Target approximation error (informational; influences defaults).

    Returns:
        np.array: Approximate centroid.
    """
    if thin is None:
        thin = d
    if burn_in is None:
        burn_in = 100 * d**2  # Heuristic based on mixing time O(d^2)
    # Compute mins and maxs
    mins = []
    maxs = []
    for v in V:
        min_val = _solve_s_optimization(S, v, d, GRB.MINIMIZE)
        max_val = _solve_s_optimization(S, v, d, GRB.MAXIMIZE)
        if min_val is None or max_val is None:
            raise ValueError("Failed to compute projections for V.")
        mins.append(min_val)
        maxs.append(max_val)
    half_spaces = list(zip(S['w'], S['x'], S['sign']))
    def feas(y):
        return feasibility_check(y, V, mins, maxs, half_spaces, tol)
    initial_p = find_point_in_S(S, d)
    if initial_p is None:
        raise ValueError("S is empty or infeasible.")
    samples, p = hit_and_run(num_samples, thin, burn_in, d, initial_p, feas, tol)
    if len(samples) == 0:
        logging.warning("No samples collected in hit-and-run.")
        return None

    centroid = np.mean(samples, axis=0)
    if return_samples:
        return centroid, samples
    return centroid


def projected_volume_update(
    delta_bar, S_t, V_t, a1, a2, d, u, 
    max_trials=None, 
    num_samples=2000, 
    thin=None, 
    burn_in=None,
    tol=1e-6, 
    rho_target=0.01,
    incentive_constant=5
    ):
    """
    Implements Algorithm 5: Projected Volume from the contextual bandit paper.
    Assumes get_centroid and diam functions are available from previous implementations.
    
    Args:
        T (int): Horizon.
        delta_bar (float): Threshold for small diameter (\bar{\omega}).
        S_t (dict): Current set {'w': list of np arrays, 'x': list of floats, 'sign': list of str ('ge' or 'le')}.
        V_t (list of np.array): Current orthonormal vectors.
        a1 (np.array): Action a_t^1.
        a2 (np.array): Action a_t^2.
        A_t (np.array): Agent's chosen action (equals a1 or a2).
        u (np.array): Agent's utility vector
        d (int): Dimension.
        max_trials (int, optional): Max random trials per addition attempt. Defaults to 100.
    
    Returns:
        dict, list: Updated S_{t+1}, V_{t+1}.
    """
    data = {}
    
    # Line 2: Compute centroid of Cyl(S_t, V_t)
    hat_s = get_centroid(S_t, V_t, d, num_samples, thin, burn_in, tol, rho_target)
    if hat_s is None:
        return S_t, V_t, data  # Return unchanged if centroid fails
    
    # Compute w_t and x_t
    diff = a1 - a2
    norm_diff = np.linalg.norm(diff)
    if norm_diff == 0:
        raise ValueError("a1 and a2 must be different")
    w_t = diff / norm_diff
    x_t = np.dot(hat_s, w_t)
    
    # Lines 4-8: Update S_{t+1} with new half-space
    S_tp1 = {'w': S_t['w'][:], 'x': S_t['x'][:], 'sign': S_t['sign'][:]}
    # if a1 @ u + incentive_constant >= incentive_constant + hat_s @ a1: # Incentive function in Line 3
    if a1 @ u >= hat_s @ a1: # Removed incentive_constant due to redundancy
        # Agent chose a1, i.e. rented out the machine
        S_tp1['w'].append(w_t.copy())
        S_tp1['x'].append(x_t)
        S_tp1['sign'].append('ge')
        data['rented'] = True
        data['profit'] = hat_s @ a1 - incentive_constant
    else:
        # Agent chose a2, i.e. passed
        S_tp1['w'].append(w_t.copy())
        S_tp1['x'].append(x_t)
        S_tp1['sign'].append('le')
        data['rented'] = False
        data['profit'] = -incentive_constant - hat_s @ a1

    # Line 9: Initialize V_{t+1}
    V_tp1 = [v.copy() for v in V_t]
    
    # Lines 10-13: Repeatedly add orthogonal v with diam <= delta_bar
    while True:
        if len(V_tp1) == d:
            break
        # Compute orthonormal basis for perpendicular subspace
        if len(V_tp1) == 0:
            perp = np.eye(d)
        else:
            V_mat = np.stack(V_tp1).T  # d x n
            _, _, Vt = np.linalg.svd(V_mat.T, full_matrices=True)
            perp = Vt[len(V_tp1):, :].T  # d x (d - n)
        added = False
        max_trials = max(1000, 100 * (d - len(V_tp1))) if max_trials is None else max_trials
        for _ in range(max_trials):
            rand_dir = np.random.randn(d - len(V_tp1))
            rand_dir /= np.linalg.norm(rand_dir) + 1e-10
            v = perp @ rand_dir
            diameter = diam(S_tp1, v, d)
            if diameter is not None and diameter <= delta_bar:
                V_tp1.append(v)
                added = True
                break
        if not added:
            break
    # while len(V_tp1) < d:
    #     # Compute perp as before
    #     if len(V_tp1) == 0:
    #         perp = np.eye(d)
    #     else:
    #         V_mat = np.stack(V_tp1).T
    #         _, _, Vt = np.linalg.svd(V_mat, full_matrices=True)  # Note: V_mat.T svd for rows
    #         perp = Vt[len(V_tp1):].T  # d x (d-n)
        
    #     # Get samples from new Cyl(S_tp1, V_tp1)
    #     _, samples = get_centroid(S_tp1, V_tp1, d, num_samples=5000, return_samples=True)  # More samples for cov accuracy
        
    #     # Project samples to L_t (perp subspace): subtract proj to V_tp1
    #     projected_samples = samples.copy()
    #     for v in V_tp1:
    #         proj = np.dot(samples, v)[:, np.newaxis] * v
    #         projected_samples -= proj
        
    #     # Sample cov of projected (mean-center first)
    #     mean_proj = np.mean(projected_samples, axis=0)
    #     centered = projected_samples - mean_proj
    #     sample_cov = np.dot(centered.T, centered) / (len(centered) - 1)
        
    #     # Eig decomp (use scipy.linalg.eigh for symmetric)
    #     eigvals, eigvecs = eigh(sample_cov)
    #     min_idx = np.argmin(eigvals)
    #     candidate_v = eigvecs[:, min_idx]
    #     candidate_v /= np.linalg.norm(candidate_v) + 1e-10
        
    #     # Compute exact diameter along candidate
    #     diameter = diam(S_tp1, candidate_v, d)
    #     if diameter is not None and diameter <= delta_bar:
    #         V_tp1.append(candidate_v.copy())
    #     else:
    #         # Certify no thinner: if even min eig dir > delta_bar, stop
    #         break
    
    
    return S_tp1, V_tp1, data

class ProjectedVolumeLearner:
    def __init__(self, T, d, centroid_params={}, incentive_constant=5, termination_rule=None):
        self.T = T
        self.d = d
        self.delta_bar = delta_bar(T, d)
        self.S_t = {'w': [], 'x': [], 'sign': []}
        self.V_t = []
        self.centroids = []
        self.centroid_params = centroid_params
        self.incentive_constant = incentive_constant
        self.termination_rule = termination_rule
        self.is_terminated = False
    
    def update(self, context, agent_utility):
        # Prioritize instance attribute over method argument
        self.S_t, self.V_t, data = projected_volume_update(
            delta_bar=self.delta_bar,
            S_t=self.S_t,
            V_t=self.V_t,
            a1=context,
            a2=np.zeros_like(context),
            d=self.d,
            u=agent_utility,
            incentive_constant=self.incentive_constant,
            **self.centroid_params
        )
        if len(data) == 0:
            self.is_terminated = True
            return data
        
        new_centroid = self.get_estimate(self.centroid_params)
        self.centroids.append(new_centroid)
        
        data['centroid'] = new_centroid
        return data
    
    def get_estimate(self, centroid_params={}):
        # if len(self.V_t) == 0:
        #     return np.zeros(self.d)
        if self.is_terminated:
            return self.centroids[-1]
        
        return get_centroid(self.S_t, self.V_t, self.d, **centroid_params)
    
    def check_termination(self, context):
        norm_context = context / np.linalg.norm(context)
        diameter = diam(self.S_t, norm_context, self.d)
        
        if self.termination_rule is None:
            done = diameter < 1 / self.T
        else:
            done = self.termination_rule(diameter)
        
        self.is_terminated = done
        return done, diameter