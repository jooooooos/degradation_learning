import numpy as np
import torch

# ================================
# Constants and Device Setup
# ================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
a, b, m = 0.1, 0.02, 2.5

# ================================
# Simulation Logic
# ================================
def simulate_machine(theta, d):
    usage = np.zeros(d)
    contexts = np.zeros((0, d))

    while True:
        x = np.random.uniform(0, 1, size=(d))
        contexts = np.vstack((contexts, x))
        current_deg = theta @ usage
        additional_deg = theta @ x
        p = np.random.random()
        hazard = 1 - np.exp(
            -(
                a * additional_deg + b * additional_deg * (3 * m**2 - 3 * m * (additional_deg + 2 * current_deg) +
                                                            additional_deg**2 + 3 * additional_deg * current_deg + 3 * current_deg**2) / 3
            )
        )
        if p <= hazard:
            return contexts
        usage += x

# ================================
# Failure Probability Function
# ================================
def failure_probability_torch(d, x):
    return 1 - torch.exp(-(
        a * x + b * x * (3 * m**2 - 3 * m * (x + 2 * d) + x**2 + 3 * x * d + 3 * d**2) / 3
    ))

# ================================
# Trajectory-Based Log-Likelihood
# ================================
def log_likelihood_trajectory_based(theta, X_trajs, Z_trajs, y_trajs):
    total_logL = 0.0
    for X, Z, y in zip(X_trajs, Z_trajs, y_trajs):
        d = Z @ theta
        x = X @ theta
        p = failure_probability_torch(d, x)
        p = torch.clamp(p, 1e-8, 1 - 1e-8)
        logL_traj = (torch.sum(torch.log(1 - p[:-1])) + torch.log(p[-1])) / len(p)
        # logL_traj = torch.sum(torch.log(1 - p[:-1])) + torch.log(p[-1])
        total_logL += logL_traj
    return total_logL

# ================================
# Optimizer Using Trajectory-Based Likelihood
# ================================
def gradient_ascent_torch_adam_trajectory(
    theta_init,
    X_trajs,
    Z_trajs,
    y_trajs,
    learning_rate=0.0001,
    num_epochs=100,
    early_stopping_tol=1e-4,
    early_stopping_min_epochs=5
):
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    prev_ll = -np.inf

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = -log_likelihood_trajectory_based(theta, X_trajs, Z_trajs, y_trajs)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ll = log_likelihood_trajectory_based(theta, X_trajs, Z_trajs, y_trajs) / len(X_trajs)
            if epoch >= early_stopping_min_epochs and torch.abs(ll - prev_ll) < early_stopping_tol:
                break
            prev_ll = ll

    return theta.detach()

# ================================
# Parallel Evaluation Function
# ================================
def evaluate_one_estimator_torch(kwargs):
    k = kwargs["k"] # number of trajectories
    d = kwargs["d"] # dimension of the context
    true_theta_np = kwargs["true_theta_np"]
    learning_rate = kwargs["learning_rate"]
    num_epochs = kwargs["num_epochs"]
    batch_size = kwargs["batch_size"]  # Not used in trajectory version
    seed = kwargs["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Simulate k trajectories
    X_trajs, Z_trajs, y_trajs = [], [], []
    for _ in range(k):
        contexts = simulate_machine(theta=true_theta_np, d=d)
        cum_contexts = np.cumsum(contexts, axis=0)
        cum_contexts = np.vstack((np.zeros(d), cum_contexts[:-1]))
        y = np.concatenate([np.zeros(len(contexts)-1), [1]])

        X_trajs.append(torch.tensor(contexts, dtype=torch.float32, device=device))
        Z_trajs.append(torch.tensor(cum_contexts, dtype=torch.float32, device=device))
        y_trajs.append(torch.tensor(y, dtype=torch.float32, device=device))

    theta_init = torch.rand(d, dtype=torch.float32, device=device)
    true_theta = torch.tensor(true_theta_np, dtype=torch.float32, device=device)

    theta_hat = gradient_ascent_torch_adam_trajectory(
        theta_init=theta_init,
        X_trajs=X_trajs,
        Z_trajs=Z_trajs,
        y_trajs=y_trajs,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    with torch.no_grad():
        l2 = torch.norm(theta_hat - true_theta, p=2).item()
        max_err = torch.max(torch.abs(theta_hat - true_theta)).item()
        avg_ll = (log_likelihood_trajectory_based(theta_hat, X_trajs, Z_trajs, y_trajs) / len(X_trajs)).item()

    return l2, max_err, avg_ll, theta_hat.cpu().numpy()
