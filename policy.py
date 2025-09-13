import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

# --- 1. Define the Neural Network for Q-Function Approximation ---

class QNetwork(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) to approximate the Q-function."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(state_dim, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, action_dim)
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# --- 2. Module for Offline Data Generation ---

class ExperienceGenerator:
    """
    Generates a dataset of experiences (s, a, r, s') by simulating the environment
    using the learned u_hat and theta_hat parameters.
    """
    def __init__(self, d, u_hat, degradation_learner, customer_generator, params):
        self.d = d
        self.u_hat = u_hat
        self.degradation_learner = degradation_learner
        self.customer_generator = customer_generator
        self.params = params

        # Action mapping
        self.ACTION_MAP = {0: 'give_price', 1: 'shutdown', 2: 'replace', 3: 'no_replace'}

    def _format_state(self, X, x, T, I, phase):
        """Standardizes the state vector representation."""
        if phase == 'arrival':
            # Full state
            return np.concatenate([X, x, [T, I, 0.0]])
        else: # departure
            # x and T are irrelevant
            return np.concatenate([X, np.zeros(self.d), [0.0, I, 1.0]])

    def _step_environment(self, state, action):
        """Simulates a single transition based on the current state and action."""
        X_prev = state[:self.d]
        # x_curr = state[self.d : 2*self.d] # Only valid at arrival
        T_curr = state[2*self.d]
        I_prev = state[2*self.d + 1]
        phase = 'arrival' if state[-1] == 0.0 else 'departure'
        
        action_name = self.ACTION_MAP[action]
        
        # --- Handle Arrival Phase ---
        if phase == 'arrival':
            if action_name == 'shutdown':
                reward = 0  # Holding cost is realized in the next step
                next_state_tuple = (X_prev, None, None, I_prev, 'departure')
            
            elif action_name == 'give_price':
                x_curr = state[self.d : 2*self.d]
                p_fail = self.degradation_learner.predict_failure_prob(X_prev, x_curr, T_curr)
                revenue = np.dot(self.u_hat, x_curr)
                
                if random.random() < p_fail:  # Machine Fails
                    reward = revenue - self.params['replacement_cost'] - self.params['failure_cost']
                    # Machine resets completely
                    next_state_tuple = (np.zeros(self.d), None, None, 0.0, 'departure')
                else:  # Machine Survives
                    reward = revenue
                    X_next = X_prev + x_curr
                    next_state_tuple = (X_next, None, None, I_prev, 'departure')
            else:
                 raise ValueError(f"Invalid action {action_name} for phase {phase}")


        # --- Handle Departure Phase ---
        elif phase == 'departure':
            customer = self.customer_generator.generate()
            tau_next = customer['interarrival_time']
            holding_reward = -self.params['holding_cost_rate'] * tau_next
            
            if action_name == 'replace':
                reward = -self.params['replacement_cost'] + holding_reward
                # Machine resets, calendar time and idle time start from tau_next
                next_state_tuple = (np.zeros(self.d), customer['context'], customer['desired_duration'], tau_next, 'arrival')

            elif action_name == 'no_replace':
                reward = holding_reward
                I_next = I_prev + tau_next
                next_state_tuple = (X_prev, customer['context'], customer['desired_duration'], I_next, 'arrival')
            else:
                raise ValueError(f"Invalid action {action_name} for phase {phase}")

        # Format the next state vector
        X_next, x_next, T_next, I_next, phase_next = next_state_tuple
        next_state = self._format_state(X_next, x_next, T_next, I_next, phase_next)
        
        return reward, next_state

    def generate(self, num_samples):
        """Generates a dataset of `num_samples` transitions."""
        dataset = []
        
        # Start with a fresh machine seeing its first customer
        X, I = np.zeros(self.d), 0.0
        customer = self.customer_generator.generate()
        state = self._format_state(X, customer['context'], customer['desired_duration'], I, 'arrival')
        
        print(f"Generating {num_samples} experience samples...")
        for _ in tqdm(range(num_samples)):
            phase_val = state[-1]
            valid_actions = [0, 1] if phase_val == 0.0 else [2, 3]
            action = random.choice(valid_actions) # Explore randomly
            
            reward, next_state = self._step_environment(state, action)
            dataset.append((state, action, reward, next_state))
            
            state = next_state
        
        return dataset
    
    
# --- 3. The Main DP Agent ---

class DPAgent:
    """Fitted Q-Iteration Agent."""
    def __init__(self, d, u_hat, degradation_learner, customer_generator, params):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_dim = 2 * d + 3
        self.action_dim = 4 # give_price, shutdown, replace, no_replace
        self.params = params
        self.gamma = params['gamma']

        # Initialize networks
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=params['learning_rate'])
        self.loss_fn = nn.MSELoss()

        # Experience generator
        self.experience_generator = ExperienceGenerator(
            d, u_hat, degradation_learner, customer_generator, params
        )

    def _get_max_q_for_valid_actions(self, states_tensor, q_values_tensor):
        """Applies action masking to find the max Q-value for valid actions."""
        max_q_values = torch.zeros(states_tensor.shape[0], device=self.device)
        
        for i in range(states_tensor.shape[0]):
            phase = states_tensor[i, -1].item()
            if phase == 0.0:  # Arrival
                valid_actions_q = q_values_tensor[i, :2]
            else:  # Departure
                valid_actions_q = q_values_tensor[i, 2:]
            max_q_values[i] = torch.max(valid_actions_q)
            
        return max_q_values

    def _evaluate_q_values(self, test_states_tensor):
        """Calculates the average Q-value for a fixed set of test states."""
        with torch.no_grad():
            q_values = self.q_network(test_states_tensor)
            max_q = self._get_max_q_for_valid_actions(test_states_tensor, q_values)
            return max_q.mean().item()

    def train(self, num_iterations, dataset_size, batch_size):
        """Main FQI training loop."""
        experience = self.experience_generator.generate(dataset_size)
        
        # Unpack and convert to tensors
        states = torch.tensor(np.array([t[0] for t in experience]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t[1] for t in experience]), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array([t[2] for t in experience]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([t[3] for t in experience]), dtype=torch.float32).to(self.device)

        # Create a fixed set of states for consistent Q-value tracking
        test_indices = np.random.choice(len(states), size=min(1000, len(states)), replace=False)
        test_states = states[test_indices]

        # History tracking
        history = {'loss': [], 'avg_q_value': []}

        print("\nStarting FQI training loop...")
        pbar = tqdm(range(num_iterations))
        for i in pbar:
            # Create a DataLoader for batching
            iteration_losses = []
            dataset = torch.utils.data.TensorDataset(states, actions, rewards, next_states)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for s_batch, a_batch, r_batch, ns_batch in loader:
                # --- Calculate the Bellman Target ---
                with torch.no_grad():
                    q_next_values = self.target_network(ns_batch)
                    max_q_next = self._get_max_q_for_valid_actions(ns_batch, q_next_values)
                    target_q = r_batch + self.gamma * max_q_next

                # --- Update the Q-Network ---
                self.optimizer.zero_grad()
                
                # Get current Q-values for the actions that were taken
                q_current_all = self.q_network(s_batch)
                q_current = q_current_all.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                # Calculate loss and update
                loss = self.loss_fn(q_current, target_q)
                loss.backward()
                self.optimizer.step()
                iteration_losses.append(loss.item())

            # Log metrics for the iteration
            avg_loss = np.mean(iteration_losses)
            avg_q = self._evaluate_q_values(test_states)
            history['loss'].append(avg_loss)
            history['avg_q_value'].append(avg_q)
            
            pbar.set_description(f"Iter {i+1}/{num_iterations} | Loss: {avg_loss:.4f} | Avg Q-Value: {avg_q:.2f}")

            # Update target network weights periodically
            if (i + 1) % self.params['target_update_freq'] == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
        print("\nTraining complete.")
        return history

    def get_policy(self):
        """Returns a function that represents the learned greedy policy."""
        def policy_fn(state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)
            
            phase = state[-1]
            if phase == 0.0: # Arrival
                valid_actions = [0, 1]
                q_subset = q_values[:2]
            else: # Departure
                valid_actions = [2, 3]
                q_subset = q_values[2:]
            
            best_action_idx = torch.argmax(q_subset).item()
            return valid_actions[best_action_idx]

        return policy_fn