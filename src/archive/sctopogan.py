import sys
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch
import graphtools

from src.Pamona.Pamona_without_projection import Pamona as Pamona_original
from umap import UMAP
from rfphate import PageRankPHATE
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from utils.utils import kernel2Dist

import os

# 1. Get the directory where THIS file (sctopogan.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Append the 'scTopoGAN' subdirectory to the system path so Python can see inside it
sc_topo_gan_path = os.path.join(current_dir, 'scTopoGAN')
if sc_topo_gan_path not in sys.path:
    sys.path.append(sc_topo_gan_path)

# 3. Add the 'src' directory itself to path just in case
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 4. Import the functions directly
# We remove the try/except block so if it fails, you see the real error immediately
from src.scTopoGAN.scTopoGAN_Functions import get_TopoAE_Embeddings, run_scTopoGAN
class scTopoGAN(Pamona_original):
    def __init__(self, 
                 mu=0.5, 
                 n_components=2, 
                 embedder='spectral', 
                 random_state=None,
                 # scTopoGAN specific parameters
                 autoencoder_model="MLPAutoencoder_PBMC", 
                 ae_arch=[50, 32, 32, 8], # Default architecture (updated dynamically in fit)
                 topo_reg_coef_a=0.5,
                 topo_reg_coef_b=3.0,
                 gan_epochs=1001,
                 gan_batch_size=512,
                 knn_neighbors=5,
                 **kwargs):
        """
        Wrapper for Manifold Alignment that uses scTopoGAN for the initial alignment
        to generate the coupling matrix T via Nearest Neighbors.

        Args:
            mu (float): Weighting factor for combining intra- and inter-dataset graphs.
            n_components (int): Dimensions for final embedding.
            embedder (str): Method for final embedding ('spectral', 'PHATE', 'UMAP').
            random_state (int): Seed.
            autoencoder_model (str): Name of the AE class in scTopoGAN (e.g. "MLPAutoencoder_PBMC").
            ae_arch (list): Architecture for the Autoencoder [input_dim, hidden..., latent].
            topo_reg_coef_a (float): Topology regularization coef for dataset A (Source).
            topo_reg_coef_b (float): Topology regularization coef for dataset B (Target).
            gan_epochs (int): Total epochs for GAN training.
            knn_neighbors (int): Neighbors for computing T from aligned data.
            **kwargs: Args passed to Pamona_original.
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.embedder = embedder
        self.n_components = n_components
        self.manual_seed = random_state
        
        # scTopoGAN configs
        self.autoencoder_model = autoencoder_model
        self.ae_arch = ae_arch
        self.topo_reg_coef_a = topo_reg_coef_a
        self.topo_reg_coef_b = topo_reg_coef_b
        self.gan_epochs = gan_epochs
        self.gan_batch_size = gan_batch_size
        self.knn_neighbors = knn_neighbors

        self.T = None
        self.W = None
        self.embedding_ = None

    def fit(self, x_a, x_b, y_a=None, y_b=None):
        """
        Runs scTopoGAN to align A -> B, computes T via KNN, and builds the joint graph W.
        """
        # 1. Formatting Inputs 
        if not isinstance(x_a, pd.DataFrame):
            df_a = pd.DataFrame(x_a)
        else:
            df_a = x_a.copy()
            
        if not isinstance(x_b, pd.DataFrame):
            df_b = pd.DataFrame(x_b)
        else:
            df_b = x_b.copy()

        self.n_a = df_a.shape[0]
        self.n_b = df_b.shape[0]

        # 2. Step 1: Get TopoAE Embeddings 
        # FIX: Dynamically update architecture based on actual input dimensions
        # to prevent "mat1 and mat2 shapes cannot be multiplied" errors.
        
        # --- Process Source (A) ---
        arch_a = self.ae_arch.copy()
        arch_a[0] = df_a.shape[1]  # Set first layer to match columns of A
        
        print(f"Training TopoAE for Source (A) with architecture: {arch_a}")
        source_latent = get_TopoAE_Embeddings(
            Manifold_Data=df_a, 
            batch_size=50, 
            autoencoder_model=self.autoencoder_model,  
            AE_arch=arch_a, 
            topology_regulariser_coefficient=self.topo_reg_coef_a, 
            initial_LR=0.001
        )

        # --- Process Target (B) ---
        arch_b = self.ae_arch.copy()
        arch_b[0] = df_b.shape[1] # Set first layer to match columns of B
        
        print(f"Training TopoAE for Target (B) with architecture: {arch_b}")
        target_latent = get_TopoAE_Embeddings(
            Manifold_Data=df_b, 
            batch_size=50, 
            autoencoder_model=self.autoencoder_model, 
            AE_arch=arch_b, 
            topology_regulariser_coefficient=self.topo_reg_coef_b, 
            initial_LR=0.001
        )

        # 3. Step 2: Manifold alignment using scTopoGAN (Source -> Target)
        print("Running scTopoGAN alignment...")
        # Note: Ensure scTopoGAN_Functions.py is patched to use device="mps" or "cpu"
        source_aligned = run_scTopoGAN(
            source_latent, 
            target_latent, 
            source_tech_name="Source", 
            target_tech_name="Target", 
            batch_size=self.gan_batch_size, 
            topology_batch_size=min(1000, self.n_a, self.n_b), 
            total_epochs=self.gan_epochs, 
            num_iterations=2, 
            checkpoint_epoch=max(1, int(self.gan_epochs/10)), 
            g_learning_rate=1e-3, 
            d_learning_rate=1e-2, 
            path_prefix="Results_TopoPamona"
        )
        
        # Convert aligned output back to numpy if it's a DataFrame or Tensor
        if isinstance(source_aligned, pd.DataFrame):
            feat_a_aligned = source_aligned.values
        elif isinstance(source_aligned, torch.Tensor):
            feat_a_aligned = source_aligned.cpu().detach().numpy()
        else:
            feat_a_aligned = source_aligned
            
        if isinstance(target_latent, pd.DataFrame):
            feat_b_latent = target_latent.values
        elif isinstance(target_latent, torch.Tensor):
            feat_b_latent = target_latent.cpu().detach().numpy()
        else:
            feat_b_latent = target_latent

        # 4. Compute T via Nearest Neighbors 
        print(f"Computing coupling matrix T using {self.knn_neighbors} nearest neighbors...")
        nbrs = NearestNeighbors(n_neighbors=self.knn_neighbors, metric='euclidean', n_jobs=-1)
        nbrs.fit(feat_b_latent)
        
        # Find neighbors in B for every point in Aligned A
        distances, indices = nbrs.kneighbors(feat_a_aligned)
        
        # Create Sparse T Matrix
        row_inds = np.repeat(np.arange(self.n_a), self.knn_neighbors)
        col_inds = indices.flatten()
        data = np.ones(len(col_inds))
        
        self.T = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(self.n_a, self.n_b)).toarray()
        
        # Normalize T to be roughly row-stochastic (soft alignment)
        self.T_norm = self.T / (self.T.max() + 1e-12)

        # 5. Build Block Matrix W 
        print("Constructing Joint Graph W...")
        # Use original data for local geometry to preserve fine structure
        prox_a = graphtools.Graph(x_a, random_state=self.manual_seed).K.toarray()
        prox_b = graphtools.Graph(x_b, random_state=self.manual_seed).K.toarray()

        # Inter-manifold connections weighted by T
        W_ab = (np.dot(prox_a, self.T_norm) + np.dot(self.T_norm, prox_b)) / 2
        W_ba = W_ab.T

        # Assemble the Block Matrix
        self.W = np.block([
            [self.mu * prox_a, (1 - self.mu) * W_ab],
            [(1 - self.mu) * W_ba, self.mu * prox_b]
        ])
        
        return self

    def fit_transform(self, x_a, x_b, y_a=None, y_b=None):
        self.fit(x_a, x_b, y_a, y_b)
    
        print(f"Computing final embedding using {self.embedder}...")
        if self.embedder == 'PHATE':
            phate_op = PageRankPHATE(
                n_components=self.n_components,
                t='auto',
                knn_dist='precomputed_affinity',
                random_state=self.manual_seed,
                n_jobs=-1,
                beta=0.5,
            )
            self.embedding_ = phate_op.fit_transform(self.W)
            return self.embedding_
    
        elif self.embedder == 'spectral':
            embedder = SpectralEmbedding(
                n_components=self.n_components,
                affinity='precomputed',
                random_state=self.manual_seed,
                n_jobs=-1,
            )
            self.embedding_ = embedder.fit_transform(self.W)
            return self.embedding_
    
        elif self.embedder == 'UMAP':
            DistM = kernel2Dist(self.W)
            self.embedding_ = UMAP(n_components=self.n_components, metric='precomputed', random_state=self.manual_seed).fit_transform(DistM)
            return self.embedding_
            
        else:
            raise ValueError(f"Unknown embedder={self.embedder}")

    def get_embeddings(self):
        return self.embedding_