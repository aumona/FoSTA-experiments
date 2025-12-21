import numpy as np
from scipy import sparse



def sugar_augmentation(X, prox, n_new, noise_scale=0.01, random_state=None, verbose=1):
        """
        Generates n_new samples using SUGAR-style manifold oversampling.
        Uses Sum-Squared-Proximity as a proxy for local density.
        """
            
        N, D = X.shape
        rng = np.random.RandomState(random_state)
        
        # ---------------------------------------------------------
        # 1. Estimate Density via "Sharpness" of Proximities
        # ---------------------------------------------------------
        # Logic: In dense regions, neighbors are consistent -> High sum of squares.
        # In sparse regions, neighbors are diffuse -> Low sum of squares.
        
        if sparse.issparse(prox):
            # Efficiently compute sum of squares for each row
            # prox.power(2) is element-wise square
            density_proxy = np.array(prox.power(2).sum(axis=1)).flatten()
        else:
            density_proxy = np.sum(prox**2, axis=1)
            
        # ---------------------------------------------------------
        # 2. Inverse Density Sampling
        # ---------------------------------------------------------
        # SUGAR logic: We want to "repair" the manifold by sampling 
        # MORE in sparse regions (low density) and LESS in dense cores.
        
        # Avoid division by zero
        density_proxy = np.maximum(density_proxy, 1e-10)
        
        # Probability ~ 1 / Density
        inv_density = 1.0 / density_proxy
        prob_seed = inv_density / inv_density.sum()
        
        # Select seed indices
        seed_indices = rng.choice(np.arange(N), size=n_new, replace=True, p=prob_seed)
        
        new_samples = np.zeros((n_new, D))
        
        # ---------------------------------------------------------
        # 3. Generate Samples (Manifold Interpolation)
        # ---------------------------------------------------------
        for k, seed_idx in enumerate(seed_indices):
            # Get diffusion neighbors
            if sparse.issparse(prox):
                row_start = prox.indptr[seed_idx]
                row_end = prox.indptr[seed_idx+1]
                neighbors = prox.indices[row_start:row_end]
                weights = prox.data[row_start:row_end]
            else:
                weights = prox[seed_idx, :]
                neighbors = np.arange(N)
            
            # Use the proximity weights themselves to pick a partner
            # This ensures we interpolate along the manifold structure
            if weights.sum() > 0:
                probs = weights / weights.sum()
            else:
                probs = np.ones(len(weights)) / len(weights)
                
            partner_idx = rng.choice(neighbors, p=probs)
            
            # Interpolate
            alpha = rng.uniform(0.1, 0.9)
            vec_diff = X[partner_idx] - X[seed_idx]
            jitter = rng.normal(0, noise_scale, size=D)
            
            new_samples[k] = X[seed_idx] + alpha * vec_diff + jitter
                    
        if verbose > 0:
            print(f"[SUGAR] Generated {n_new} samples (Metric: Sum-Sq-Prox).")
            
        return new_samples