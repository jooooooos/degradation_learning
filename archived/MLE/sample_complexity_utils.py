import numpy as np
import torch

# Define constants
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
a, b, m = 0.1, 0.02, 2.5

# Simulation logic
def simulate_machine(theta, d):
    usage = np.zeros(d)
    contexts = np.zeros((0, d))

    while True:
        x = np.random.uniform(0, 1, size=(d))
        # print(theta.shape, x.shape, contexts.shape)
        contexts = np.vstack((contexts, x))
        current_deg = theta @ usage
        # print(theta.shape, x.shape, contexts.shape, usage.shape)
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

# Log-likelihood + optimizer
def failure_probability_torch(d, x):
    return 1 - torch.exp(-(
        a * x + b * x * (3 * m**2 - 3 * m * (x + 2 * d) + x**2 + 3 * x * d + 3 * d**2) / 3
    ))

def log_likelihood_torch(theta, X, Z, y):
    d = Z @ theta
    x = X @ theta
    p = failure_probability_torch(d, x)
    p = torch.clamp(p, 1e-8, 1 - 1e-8)
    return torch.sum(y * torch.log(p) + (1 - y) * torch.log(1 - p))

def gradient_ascent_torch_adam(
    theta_init,
    X,
    Z,
    y,
    learning_rate=0.0001,
    batch_size=32,
    num_epochs=100,
    early_stopping_tol=1e-4,
    early_stopping_min_epochs=5
):
    theta = theta_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    n = X.shape[0]
    prev_ll = -np.inf

    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        X_shuffled, Z_shuffled, y_shuffled = X[perm], Z[perm], y[perm]

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_X = X_shuffled[start:end]
            batch_Z = Z_shuffled[start:end]
            batch_y = y_shuffled[start:end]

            optimizer.zero_grad()
            loss = -log_likelihood_torch(theta, batch_X, batch_Z, batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            ll = log_likelihood_torch(theta, X, Z, y) / n
            if epoch >= early_stopping_min_epochs and torch.abs(ll - prev_ll) < early_stopping_tol:
                break
            prev_ll = ll

    return theta.detach()

# The parallel evaluation function
def evaluate_one_estimator_torch(kwargs):
    k = kwargs["k"]
    d = kwargs["d"]
    true_theta_np = kwargs["true_theta_np"]
    learning_rate = kwargs["learning_rate"]
    num_epochs = kwargs["num_epochs"]
    batch_size = kwargs["batch_size"]
    seed = kwargs["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # simulate
    X, Z, y = [], [], []
    for _ in range(k):
        contexts = simulate_machine(theta=true_theta_np, d=d)
        cum_contexts = np.cumsum(contexts, axis=0)
        cum_contexts = np.vstack((np.zeros(d), cum_contexts[:-1]))
        X.append(contexts)
        Z.append(cum_contexts)
        y.append(np.concatenate([np.zeros(len(contexts)-1), [1]]))

    X_tensor = torch.tensor(np.vstack(X), dtype=torch.float32, device=device)
    Z_tensor = torch.tensor(np.vstack(Z), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.concatenate(y), dtype=torch.float32, device=device)

    theta_init = torch.rand(d, dtype=torch.float32, device=device)
    true_theta = torch.tensor(true_theta_np, dtype=torch.float32, device=device)

    theta_hat = gradient_ascent_torch_adam(
        theta_init=theta_init,
        X=X_tensor,
        Z=Z_tensor,
        y=y_tensor,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    with torch.no_grad():
        l2 = torch.norm(theta_hat - true_theta, p=2).item()
        max_err = torch.max(torch.abs(theta_hat - true_theta)).item()
        ll = (log_likelihood_torch(theta_hat, X_tensor, Z_tensor, y_tensor) / len(X_tensor)).item()

    return l2, max_err, ll
