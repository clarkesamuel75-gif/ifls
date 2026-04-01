import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import pandas as pd
from scipy.special import expit
import math
from itertools import combinations
import time
from collections import deque


class FederatedEnsemble:
    """
    Ensemble that aggregates predictions from client Random Forest model and provides access to individual client logits for Shapley value calculation
    
    Privacy: This class runs on the server
    - server holds the trained models from clients
    - Server uses its own validation data (X_val, y_val) for evaluation
    - Clients never see the validation data
    - get_base_logits() queries models on server-held validation data
    """

    def __init__(self, models, data_sizes, task = 'classifcation'):
        """
        Args:
           models: list of trained client Random Forest models
           data_sizes: list of client dataset sizes, used to compute aggregation weights
           task: 'regresion' or 'classification'
        
        """
        self.models = models
        self.weights = np.array(data_sizes)/sum(data_sizes)
        self.task = task

    def get_base_logits(self, X):
        """
        Extract per-client output-space representation for a given input X.
        return logits for classification and raw predictions for regression

        Args:
            X: input features, shape (n_samples, n_features)
        Returns:
            numpy array of shape (n_samples, n_clients)
        """
        if self.task == 'classification':
            client_probs = np.column_stack([
                model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1] == 2
                else model.predict_proba(X)
                for model in self.models
            ])
            # Convert to logits for binary classification
            eps = 1e-15
            client_probs = np.clip(client_probs, eps, 1-eps)
            return np.log(client_probs/(1-client_probs))
        elif self.task == 'regression':
           return np.column_stack([model.predict(X) for model in self.models])
        else:
            raise ValueError(f"Task {self.task} not supported")
        
    def predict_proba(self, X):
        """
        Compute aggregated predicted probabilities via weighted averaging of client logits/predictions

        Args:
            X: input features, shape (n_samples, n_features)
        Returns:
           if classification: array of shape (n_samples, 2) with class probs
           if regression: array of shape (n_samples, )
        """
        client_logits = self.get_base_logits(X)
        global_logit = np.average(client_logits, axis = 1, weights = self.weights)
        if self.task == 'classification':
            prob_1 = expit(global_logit)
            return np.vstacck([1-prob_1, prob_1]).T
        else:
            return global_logit
    
    
    def predict(self, X):
        """
        Return predictions (class labels for classificaiton, values for regression)

        Args:
           X: input featuresm shape (n_samples, n_features)
        Returns:
            array of predicted class albels for classification and predicted values for regression
        """

        if self.task == 'classification':
            proba = self.predict_proba(X)
            return np.argmax(proba, axis = 1)
        else:
            return self.predict_proba

     

def ClientUpdate_TreeBased(client_id, X_local, y_local, task = 'classification', **model_kwargs):
    """
    Train a tree-based model locally

    Privacy: this function runs on each client in parallel.
    - X_local, y_local never leave the client
    - Only the trained model shared with server

    Parameters:
       client_id: Identifier for the client
       X_local: Local training features
       y_local: local training labels
       task: 'classification' or 'regression'
       **model_kwargs: additional keyword arguments passed to the model constructor
    
    Returns:
       model: trained Random Forest model (sent to server)
    """

    if 'random_state' not in model_kwargs:
        model_kwargs['random_state'] = client_id
    
    if task == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_kwargs)
    
    elif task == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_kwargs)
    
    else:
        raise ValueError(f"Task must be 'classification' or 'regression. Got {task}")
    
    model.fit(X_local, y_local)
    
    return model

def FedAvg_TreeBased(client_models, data_sizes, T, C, K, X_train_clients, y_train_clients, task = 'classification', model_kwargs = None,  X_test = None):
    """
    Federated Averaging for tree-based models via logit/prediction averaging
    (Horizontal Federated Learning setup)

    Privacy Notes:
    - This simulates the FL process (in practise, clients would be separate machines)
    - X_train_cient, y_train_clients represnt data that stays on clients 
    - Only trained models are communicated to the sever
    - Server neber accesses client training data

    Args:
       client_models: list of K client models
       data_sizes: list of dataset sizes for each client
       T: number of communication rounds
       C: Fraction of clients selected per round
       K: Total number of clients
       X_train_clients: List of training features for each client 
       y_train_clients: List of training labels for each client
       task: 'classification' or 'regression'
       model_kwargs: dict, additional parameters for RandomForest
       X_test: Test set for evaluation
    
    Returns:
       aggregated_model: A federated ensemble that averages predictions
    """
    if model_kwargs is None:
        model_kwargs = {}


    # T is set to 1 throughout all experiments; tree-based FL has no iterative weight updates
    for t in range(T):
        # Select a subset of the clients
        m = max(int(C * K), 1)
        S_t = np.random.choice(range(K), m, replace = False)

        # Train local models
        local_models = []
        selected_sizes = []

        for k in S_t:
            model_k = ClientUpdate_TreeBased(k, 
                                             X_train_clients[k], 
                                             y_train_clients[k],
                                             task = task,
                                             **model_kwargs)
            local_models.append(model_k)
            selected_sizes.append(data_sizes[k])
    return FederatedEnsemble(local_models, selected_sizes, task = task)


def exact_shapley(client_logits, y_val, data_sizes, task = 'classification'):
    """
    Compute exact Shapley values by enumerating all 2^K coalitions

    Utility is negative log-loss for classification or negative MSE for regression weighted by gloval client data proportions.

    Args:
       client_logits: array of shape (K, n_samples), pre-computed per-client logits
       y_val: validation labels
       data_sizes: list of client dataset sizes for weighting
       task: 'classification' or 'regression'
    Returns:
       phi: array of shape (K,) of exact Shapley values
    """

    n = client_logits.shape[0]

    if n > 20:
        raise ValueError(f"Exact Shapley Calculation is infeasible for n= {n}")
    
    weights = np.array(data_sizes)
    total_weight = np.sum(weights)

    def get_utility(indices):
        """Utility function: Negative Log Loss, the higher the better"""
        if len(indices) == 0:
            if task == 'classification':
                return -log_loss(y_val, np.full(len(y_val), 0.5), labels = [0,1])
            else:
                return -np.var(y_val)
            
        sub_logits = client_logits[indices,:]
        sub_weights = weights[indices]

        total_ensemble_weight = np.sum(weights)

        sub_logits = np.atleast_2d(sub_logits)
        weighted_sum_logits = sub_weights @ sub_logits
        avg_logit = weighted_sum_logits / total_ensemble_weight

        if task == 'classification':
            probs = expit(avg_logit)
        # Ensure probs is 1D (samples,)
            probs = probs.ravel() 
            return -log_loss(y_val, probs, labels=[0,1])
        else:
            return -np.mean((avg_logit.ravel() - y_val)**2)
        
    coalition_map = {}
    for size in range(n+1):
        for coalition in combinations(range(n), size):
            coalition_map[coalition] = get_utility(list(coalition))

    phi = np.zeros(shape = n)
    for i in range(n):
        shapley_value = 0.0
        for s in range(n):
            indices_without_i = [x for x in range(n) if x!= i]
            for S in combinations(indices_without_i, s):
                S_tuple = tuple(sorted(S))
                S_with_i = tuple(sorted(list(S) + [i]))

                marginal = coalition_map[S_with_i] - coalition_map[S_tuple]
                weight = (math.factorial(s) * math.factorial(n -s - 1))/math.factorial(n)
                shapley_value += weight *marginal

        phi[i] = shapley_value
    return phi


def get_client_logits(client_models, X_val, task='classification'):
    """
    Pre-compute per-client output-space logits on the validation set and caches results 

    Args:
        client_models: list of trained client models
        X_val: validation features
        task: 'classification' or 'regression'
    Returns:
        array of shape (K, n_samples)
    """
    logits = []
    for model in client_models:
        if task == 'classification':
            p = model.predict_proba(X_val)
            p_pos = p[:, 1] if p.shape[1] == 2 else (np.ones(len(X_val)) if model.classes_[0] == 1 else np.zeros(len(X_val)))
            # Convert to logit
            l = np.log(np.clip(p_pos, 1e-15, 1-1e-15) / (1 - np.clip(p_pos, 1e-15, 1-1e-15)))
            logits.append(l)
        else:
            logits.append(model.predict(X_val))
    return np.array(logits)



def GTG_Shapley(client_logits, y_val, data_sizes,
                max_iter=100, task='classification', convergence_threshold=0.05, m=10):
    
    """
    Adapted form of method from Liu et al. 2022.
    Compute Shapley values via the Group Testing Gradient (GTG) approximation.
    Uses truncated Monte Carlo permutation sampling with an early stopping criterion
    based on relative change across the last m iterations.

    Args:
        client_logits: array of shape (K, n_samples)
        y_val: validation labels
        data_sizes: list of client dataset sizes
        max_iter: maximum number of permutation samples
        task: 'classification' or 'regression'
        convergence_threshold: relative change threshold for early stopping (delta in paper)
        m: window size for convergence check
    Returns:
        phi: array of shape (K,) containing approximate Shapley values
    """

    n = client_logits.shape[0]
    phi = np.zeros(n)
    phi_history = deque(maxlen=m + 2)  
    total_N = sum(data_sizes)

    # --- Precompute all model outputs once ---
    model_outputs = []

    for i in range(n):
        weight = data_sizes[i] /total_N
        model_outputs.append(client_logits[i] * weight)

    def eval_func_incremental(accumulated_output, new_idx):
        """Add one model's contribution to an existing accumulated output."""
        return accumulated_output + model_outputs[new_idx]

    def score(accumulated_output, is_empty=False):
        if is_empty:
            if task == 'classification':
                return -log_loss(y_val, np.full(len(y_val), 0.5), labels=[0, 1])
            else:
                return -mean_squared_error(y_val, np.zeros_like(y_val))
        if task == 'classification':
            return -log_loss(y_val, expit(accumulated_output), labels=[0, 1])
        else:
            return -mean_squared_error(y_val, accumulated_output)

    v_0 = score(None, is_empty=True)

    for k in range(1, max_iter + 1):
        pi_k = np.random.permutation(n)
        v_prev = v_0
        accumulated = np.zeros(len(y_val))  # Reset coalition output each iteration

        for j in range(n):
            accumulated = eval_func_incremental(accumulated, pi_k[j])  # O(1) update
            v_curr = score(accumulated)
            phi[pi_k[j]] = ((k - 1) / k) * phi[pi_k[j]] + (1 / k) * (v_curr - v_prev)
            v_prev = v_curr

        phi_history.append(phi.copy())

        if k > m:
            phi_current = phi_history[-1]  # Pull out of loop
            convergence_sum = sum(
                np.sum(np.where(
                    np.isfinite(np.abs(phi_current - phi_history[-lag - 1]) / np.abs(phi_current)),
                    np.abs(phi_current - phi_history[-lag - 1]) / np.abs(phi_current),
                    0
                ))
                for lag in range(1, m + 1)
            )
            convergence_value = convergence_sum / (n * m)

            if convergence_value < convergence_threshold:
                print(f"Converged at iteration {k} (convergence value: {convergence_value:.6f})")
                break
    else:
        print(f"Max iterations ({max_iter}) reached without convergence")

    return phi                                       

def permutation_shapley_fixed_weight(client_models, X_val, y_val, client_data_sizes, M=500, seed=42, task = 'classification'):
    """
    Compute Shapley values via Monte Carlo permutation sampling with fixed data-size weights.
    Uses M random permutations to estimate marginal contributions.

    Args:
        client_models: list of trained client models
        X_val: validation features
        y_val: validation labels
        client_data_sizes: list of client dataset sizes for weighting
        M: number of permutation samples
        seed: random seed for reproducibility
        task: 'classification' or 'regression'
    Returns:
        array of shape (K,) of approximate Shapley values
    """
    n_val = len(y_val)
    K = len(client_models)
    total_N = sum(client_data_sizes)
    
    client_outputs = []

    for model in client_models:
        if task == 'classification':
            p = model.predict_proba(X_val)
            p_pos = p[:, 1] if p.shape[1] == 2 else (np.ones(n_val) if model.classes_[0] == 1 else np.zeros(n_val))
            client_outputs.append(np.log(np.clip(p_pos, 1e-15, 1-1e-15) / (1 - np.clip(p_pos, 1e-15, 1-1e-15))))
        else:
            client_outputs.append(model.predict(X_val))
                
    client_outputs = np.array(client_outputs)
    
    if task == 'classifcation':
        v_empty = -log_loss(y_val, np.full(n_val, 0.5), labels=[0, 1])
    else:
        v_empty = -mean_squared_error(y_val, np.zeros(n_val))
    
    rng = np.random.default_rng(seed)
    shapley_values = np.zeros(K)

    for m in range(M):
        perm = rng.permutation(K)
        prev_utility = v_empty
        current_sum = np.zeros(n_val)
        
        for k in perm:
            current_sum += client_outputs[k] * (client_data_sizes[k]/total_N)
            if task == 'classification':
                new_utility = -log_loss(y_val, expit(current_sum), labels=[0, 1])
            else:
                new_utility = -mean_squared_error(y_val, current_sum)
            shapley_values[k] += (new_utility - prev_utility)
            prev_utility = new_utility
    
    return shapley_values / M        





def IFLS(X_val, y_val, fed_ensemble, task = 'classification', M = 20):
    """
    Compute linearised loss-based Shapley approximations for a trained federated ensemble.

    The folliwng runs on the Server in practise:
    - X_val, y_val are server-held validation data
    - Claints never see this validation data
    - server queries client models for predictions on X_val
    - This is standard FL contribution evaluation practise

    Parameters:
        X_val: validation features (server-held)
        y_val: validation labels (server-held)
        fed_ensemble: trained FederatedEnsemble instance
        task: 'classification' or 'regression'
        M: number of quadrature points for gradient approximation
    Returns:
        phi_mean: array of shape (K,) containing IFLS Shapley values
    """

    weights = fed_ensemble.weights
    K = len(fed_ensemble.models)
    total_weight = np.sum(weights)


    logits = fed_ensemble.get_base_logits(X_val)
    if logits.shape[0] == K: # If it's (K, Samples), flip it
        logits = logits.T


    global_logit = (logits * weights[None, :]).sum(axis = 1)/total_weight
    contrib = (logits * weights[None,:])/total_weight
    y = np.asarray(y_val).flatten()

    grad_accum = np.zeros_like(global_logit, dtype = float)

    if task == 'classification':
        from scipy.special import expit
        for m in range(1, M+1):
            alpha = (m-0.5)/M
            probs = expit(alpha * global_logit)
            grad_accum += (probs- y)
        grad_mean = grad_accum/M

    elif task == 'regression':
        for m in range(1, M+1):
            alpha = (m-0.5)/M
            grad_accum += (alpha *global_logit - y)
        grad_mean  = grad_accum/M
    else:
        raise ValueError(f"Task must be 'classification' or 'regression'.")
    
    phi_mean = -(contrib * grad_mean[:, np.newaxis]).mean(axis = 0)
    return phi_mean

def make_clients_pure_label_skew(X, y, K=10, samples_per_client= None, alpha=0.5, seed = 42):
    """
    Partition data across K clients using Dirichlet label skew.
    Each client's class proportions are drawn from a symmetric Dirichlet(alpha),
    producing heterogeneous distributions. Lower alpha = greater skew.

    Args:
        X, y: full training data
        K: number of clients
        samples_per_client: int, list of ints, or None (defaults to equal split)
        alpha: Dirichlet concentration parameter
        seed: random seed
    Returns:
        clients: list of (X_k, y_k) tuples
        client_sizes: list of dataset sizes
    """
    np.random.seed(seed)
    if samples_per_client is None:
        samples_per_client = [len(y) // K] *K
    elif isinstance(samples_per_client, int):
        samples_per_client = [samples_per_client] * K
    elif isinstance(samples_per_client, list):
        assert len(samples_per_client) == K, f"samples per client list must have length K = {K}"
    else:
        raise ValueError("samples_per_client must be int, lsit, or None")

    # 1. Group indices by class
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    # 2. Determine class proportions for each client using Dirichlet
    # This ensures each client has a different "mix"
    class_proportions = np.random.dirichlet([alpha, alpha], K) 
    
    clients = []
    client_sizes = []

    for k in range(K):
        n_samples = samples_per_client[k]
        # Calculate how many of each class this client needs to reach samples_per_client
        n1 = int(n_samples * class_proportions[k, 1])
        n0 = n_samples - n1
        
        # Draw indices (with replacement if dataset is small, or without if large)
        selected_idx0 = np.random.choice(idx0, n0, replace=False)
        selected_idx1 = np.random.choice(idx1, n1, replace=False)
        
        inds = np.concatenate([selected_idx0, selected_idx1])
        np.random.shuffle(inds)

        clients.append((X[inds], y[inds]))
        n = len(y[inds])
        client_sizes.append(n)
        
        frac1 = float(y[inds].mean()) if n > 0  else float("nan")
        print(f"Client {k:2d}: n={n:4d}, p(y=1)={frac1:.3f}")
        
    return clients, client_sizes


def make_clients_size_skew(X, y, K = 10, seed = 0):
    """
    Partition data across K=10 clients with tiered quantity skew,
    following the setup of Liu et al. (2022). Clients receive unequal
    dataset sizes according to fixed proportions, with random shuffling
    before assignment.

    Args:
        X, y: full training data
        K: number of clients (must be 10)
        seed: random seed
    Returns:
       clients: list of (X_k, y_k) tuples
        data_sizes: list of dataset sizes
    """

    assert K == 10
    rng = np.random.seed(seed)
    
    # Per-client proportions mirroring GTG-Shapley
    proportions = np.array([
        0.10, 0.10,
        0.15, 0.15,
        0.20, 0.20,
        0.25, 0.25,
        0.30, 0.30
    ])
    # Already sums to 1.0, but normalise defensively
    proportions /= proportions.sum()
    
    n = len(X)
    sizes = np.floor(proportions * n).astype(int)
    sizes[-1] += n - sizes.sum()
    
    idx = np.random.permutation(n)
    X_shuffled, y_shuffled = X[idx], y[idx]
    
    clients, data_sizes = [], []
    start = 0
    for size in sizes:
        clients.append((X_shuffled[start:start + size], y_shuffled[start:start + size]))
        data_sizes.append(size)
        start += size
    
    return clients, data_sizes




def load_data(task = 'classification'):
    """
    Fetch and preprocess the Adult Income dataset from OpenML.
    Categorical features are label-encoded. Data is split into
    train (60%), validation (20%), and test (20%) sets.

    Args:
        task: 'classification' (binary income label) or 'regression' (float label)
    Returns:
        X_train, X_val, y_train, y_val
    """
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    X = adult.data.copy()
    if task == 'classification':
        y = (adult.target == '>50K').astype(int).values
        stratify_y = y
    else:
        y = (adult.target == '>50K').astype(float).values
        stratify_y = None

    cat_cols = X.select_dtypes(include=['category','object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    X = X.values
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )
    second_strat = y_trainval if task == 'classification' else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=second_strat
    )
    return X_train, X_val, y_train, y_val


from scipy.stats import spearmanr, pearsonr

def calculate_metrics(target, reference, prefix):
    """
    Compute alignment metrics between a target and reference Shapley vector.
    Returns Spearman rank correlation, Pearson correlation, R^2,
    cosine distance, Euclidean distance, and max absolute difference.

    Args:
        target: Shapley values from the method being evaluated
        reference: ground-truth Shapley values (typically exact)
        prefix: string prefix for result dict keys (e.g. 'ifls', 'gtg', 'perm')
    Returns:
        dict of metric_name -> value
    """
    rho, _ = spearmanr(target, reference)
    r, _ = pearsonr(target, reference)

    norm_t = np.linalg.norm(target)
    norm_r = np.linalg.norm(reference)

    if norm_t == 0 or norm_r == 0:
        cos_dist = 1.0
    else:
        cos_dist = 1-np.dot(target, reference)/(norm_t * norm_r)

    euclidean = np.sqrt(np.sum((target-reference) ** 2))
    max_diff = np.max(np.abs(target-reference))

    return{
        f'{prefix}_spearman':rho,
        f'{prefix}_pearson':r,
        f'{prefix}_r2': r**2,
        f'{prefix}_cosine': cos_dist,
        f'{prefix}_euclidean': euclidean,
        f'{prefix}_max_diff': max_diff
    }

def run_experiment(K, alpha, seed, X_train, y_train, X_val, y_val, task = 'classification', scenario = 1):
    """
    Run a single experimental trial: partition data, train federated ensemble,
    compute all four Shapley variants (Exact, IFLS, GTG, Permutation),
    and return alignment metrics and runtimes.

    Args:
        K: number of clients
        alpha: Dirichlet concentration parameter (used in scenario 1)
        seed: random seed
        X_train, y_train: training data
        X_val, y_val: server-held validation data
        task: 'classification' or 'regression'
        scenario: 1 for label skew, 2 for size skew
    Returns:
        results: dict containing K, alpha, seed, scenario, timing, and metric values
    """
    from scipy.special import expit
    import numpy as np
    print(f"[Start] K={K}, Alpha={alpha}, Seed={seed}")
    
    if scenario == 1:
        # Create skewed clients
        clients, data_sizes = make_clients_pure_label_skew(
            X = X_train, y = y_train, K = K, alpha = alpha, seed = seed
        )
    elif scenario == 2:
        clients, data_sizes = make_clients_size_skew(
            X=X_train, y=y_train, K=K, seed=seed
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 1 or 2.")
    X_train_c = [client[0] for client in clients]
    y_train_c = [client[1] for client in clients]

    # Train Federated Ensemble
    fed_ensemble = FedAvg_TreeBased(
        client_models = None, 
        data_sizes = data_sizes, 
        T = 1, C = 1, K=K,
        X_train_clients=X_train_c, y_train_clients = y_train_c,
        task = task, 
        model_kwargs = {'n_estimators': 100, 'random_state': seed}
    )

    client_logits = get_client_logits(fed_ensemble.models, X_val,task = task)

    # Run all Shapley variants
    results = {'K': K, 'alpha': alpha if scenario == 1 else 'N/A', 'seed':seed, 'scenario': scenario}

    start = time.time()
    phi_exact = exact_shapley(client_logits,y_val, data_sizes, task = task)
    results['time_exact'] = time.time()- start

    start = time.time()
    phi_ifls = IFLS(X_val, y_val, fed_ensemble, task = task, M = 20)
    results['time_ifls'] = time.time() - start
    results.update(calculate_metrics(phi_ifls, phi_exact, 'ifls'))


    start = time.time()
    phi_gtg = GTG_Shapley(client_logits, y_val, data_sizes, max_iter = 100, task = task)
    results['time_gtg'] = time.time() - start
    results.update(calculate_metrics(phi_gtg, phi_exact, 'gtg'))

    start = time.time() 
    phi_perm = permutation_shapley_fixed_weight(fed_ensemble.models, X_val, y_val, data_sizes, M = 500, task = task)
    results['time_perm'] = time.time() - start
    results.update(calculate_metrics(phi_perm, phi_exact, 'perm'))
    
    return results
