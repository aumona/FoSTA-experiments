import numpy as np
import ot  # Python Optimal Transport
from rfgap import RFGAP
from phate import PHATE
from scipy import sparse
from scipy.spatial.distance import cdist

class RFMALI(object):
    """
    RF-MALI: Semi-Supervised Manifold Alignment using RFGAP label bridge.
    
    This method uses Random Forest Proximities to estimate
    posterior probabilities for each sample. These posteriors form a C-dimensional
    feature space used to align the domains via Optimal Transport.
    """

    def __init__(self, mu=0.5, n_components=2, knn_dist='precomputed_affinity', random_state=None, **kwargs):
        """
        Initialize the alignment model.
        
        Parameters
        ------------
        mu : float
            Weight for the diagonal blocks in the final fusion.
        n_components : int
            Number of PHATE dimensions.
        kwargs :
            Other arguments passed to the rfphate.RFPHATE constructor.
        """
        self.mu = mu
        self.n_components = n_components
        self.knn_dist = knn_dist
        self.random_state = random_state
        self.embedder_params = kwargs
        
        self.T = None  # Coupling matrix
        self.W_combined = None  # Combined affinity matrix
        self.z_a = None
        self.z_b = None
        self.n_a = 0
        self.n_b = 0
        self.embedding_ = None
        self.rfgap_a = None
        self.rfgap_b = None

    def fit(self, x_a, y_a, x_b, y_b):
        """
        Fits the alignment model.
        """
        print("Fitting RFGAP on Domain A...")
        self.n_a = x_a.shape[0]
        y_a = y_a.ravel() 
        # Note: Using RFGAP to generate the Random Forest and Proximities
        self.rfgap_a = RFGAP(y=y_a, random_state=self.random_state, non_zero_diagonal=True)
        self.rfgap_a.fit(x_a, y_a)
        prox_a = self.rfgap_a.get_proximities() # Get sparse proximity matrix
        
        print("Fitting RFGAP on Domain B...")
        self.n_b = x_b.shape[0]
        y_b = y_b.ravel()
        self.rfgap_b = RFGAP(y=y_b, random_state=self.random_state, non_zero_diagonal=True)
        self.rfgap_b.fit(x_b, y_b)
        prox_b = self.rfgap_b.get_proximities() # Get sparse proximity matrix

        # Determine common label space
        labels_a = np.unique(y_a)
        labels_b = np.unique(y_b)
        all_labels = np.union1d(labels_a, labels_b)

        print("Building Manifold-Smoothed Posteriors (RFGAP)...")
        post_a = self._get_rfgap_posteriors(prox_a, y_a, all_labels)   # C-dim vectors needed for cross similarity
        post_b = self._get_rfgap_posteriors(prox_b, y_b, all_labels)    # C-dim vectors needed for cross similarity
        
        print("Computing Optimal Transport Plan...")
        # Uniform weights
        p_a = np.ones(self.n_a, dtype=float) / self.n_a  # probability distributions
        p_b = np.ones(self.n_b, dtype=float) / self.n_b # probability distributions

        # Cost Matrix: Jensen-Shannon Divergence between posterior distributions
        # using scipy.spatial.distance.cdist because ot.dist can be restrictive
        cost_matrix = cdist(post_a, post_b, metric='jensenshannon')
        
        # Square the JS metric if desired, though standard JS is a metric
        # cost_matrix = cost_matrix ** 2 
        
        self.T = ot.emd(p_a, p_b, cost_matrix)  #TODO: Look at Sinkhorn / Hierarchical OT for scalability
        #TODO Look at coarse graining (MSPHATE) for refinement of OT
        
        # Convert T to sparse for memory efficiency in fusion
        T_sparse = sparse.csr_matrix(self.T)
        
        #TODO Check Normalization of T. Not sure here
        # Normalize T for the fusion step (Scale up to match proximity magnitudes)
        # Typically we want row-sums of T to look like row-sums of Prox
        # A simple heuristic is normalizing by max or creating a transition matrix
        # Here we follow the MALI approach of just using T directly or normalized
        T_sparse = T_sparse / T_sparse.max() if T_sparse.max() > 0 else T_sparse

        print("Building joint affinity matrix (MALI-style fusion)...")
        
        # Ensure proximities are symmetric
        prox_a = (prox_a + prox_a.T) / 2
        prox_b = (prox_b + prox_b.T) / 2
        
        # Fusion: Transport Proximities
        # W_ab = (P_a * T + T * P_b) / 2
        W_ab = (prox_a.dot(T_sparse) + T_sparse.dot(prox_b)) / 2
        W_ba = W_ab.T

        #TODO Handle partially labeled target domain: we cannot compute unlabeled-unlabeled RFGAP affinities directly.
        #NOTE Suggestion: Use only labeled points as anchors, then project via Landmark PHATE
        # Build the giant block matrix
        # [ mu * P_a      (1-mu) * W_ab ]
        # [ (1-mu) * W_ba   mu * P_b    ]
        self.W_combined = sparse.bmat(
            [
                [self.mu * prox_a, (1-self.mu) * W_ab],
                [(1-self.mu) * W_ba, self.mu * prox_b]
            ],
            format="csr"
        )
        
        print("Model fit complete.")
        return self

    def fit_transform(self, x_a=None, y_a=None, x_b=None, y_b=None):
        """
        Embeds the joint kernel using PHATE.
        """
        if x_a is not None:
            self.fit(x_a, y_a, x_b, y_b)

        if self.W_combined is None:
            raise RuntimeError("You must call fit() before fit_transform().")

        print("Computing joint PHATE embedding from W_combined...")
        
        phate_op = PHATE(
            n_components=self.n_components,
            knn_dist=self.knn_dist,
            random_state=self.random_state
        )
        
        # Since W_combined is a precomputed affinity matrix, we pass it directly
        self.embedding_ = phate_op.fit_transform(self.W_combined)

        print("Joint embedding complete.")
        return self.embedding_