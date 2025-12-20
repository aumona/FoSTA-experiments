##### DOT NOT USE; ASSUMES SAME FEATURES OR SHARED FEATURES #####
#################################################################

import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch

class scANVI:
    def __init__(self, n_components=30, n_layers=2, n_hidden=128, max_epochs=100, 
                 unlabeled_token="Unknown", random_state=None, use_gpu=True):
        
        self.n_latent = n_components
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.unlabeled_token = unlabeled_token
        self.random_state = random_state
        
        # --- GPU Detection Logic ---
        # Checks if we can use NVIDIA (CUDA) or Apple Silicon (MPS)
        self.accelerator = "cpu"
        self.devices = "auto"

        if use_gpu:
            if torch.cuda.is_available():
                self.accelerator = "gpu"
            elif torch.backends.mps.is_available():
                self.accelerator = "mps"
            else:
                self.accelerator = "cpu"
        
        self.model_ = None
        self.adata_ = None

    def _process_labels(self, y):
        # HELPER: Normalizes labels.
        # Crucial for Semi-Supervised: We must explicitly mark missing labels 
        # with a specific string (e.g., "Unknown") so scANVI knows to ignore them 
        # for the classification loss but use them for the geometric loss.
        y = np.array(y, dtype=object)
        is_missing = pd.isnull(y) | (y == -1) | (y == "-1") | (y == None)
        y_str = y.astype(str)
        y_str[is_missing] = self.unlabeled_token
        return y_str

    def fit(self, x_a, x_b, y_a, y_b):
        # 0. Reproducibility
        if self.random_state is not None:
            scvi.settings.seed = self.random_state
        
        # 1. Data Prep
        # Wrap numpy arrays into AnnData objects (required by scvi-tools)
        adata_a = ad.AnnData(X=x_a)
        adata_a.obs['batch'] = 'source'
        adata_a.obs['cell_type'] = self._process_labels(y_a)
        
        adata_b = ad.AnnData(X=x_b)
        adata_b.obs['batch'] = 'target'
        adata_b.obs['cell_type'] = self._process_labels(y_b)
        
        # Concatenate into one object. 'batch_indices' tracks which dataset came from where.
        self.adata_ = ad.concat([adata_a, adata_b], label="batch_indices", index_unique="-")
        
        # Register the data columns so scVI knows where to find batches and labels
        scvi.model.SCVI.setup_anndata(
            self.adata_, 
            batch_key="batch", 
            labels_key="cell_type"
        )
        
        # --- STEP 1: UNSUPERVISED PRE-TRAINING (scVI) ---
        # We initialize a standard VAE. It doesn't use the labels for training yet.
        vae = scvi.model.SCVI(
            self.adata_, 
            n_layers=self.n_layers, 
            n_hidden=self.n_hidden, 
            n_latent=self.n_latent,
            gene_likelihood="normal" 
        )
        
        # Train the VAE to learn the data geometry (manifold)
        vae.train(
            max_epochs=min(self.max_epochs, 100), 
            accelerator=self.accelerator, 
            devices=1 if self.accelerator != "cpu" else "auto"
        )
        
        # --- STEP 2: SEMI-SUPERVISED REFINEMENT (scANVI) ---
        # 'from_scvi_model' is the magic function.
        # It takes the trained weights from 'vae' (Step 1) and copies them into a new 'scANVI' model.
        # This means scANVI starts already knowing the shape of the data.
        self.model_ = scvi.model.SCANVI.from_scvi_model(
            vae, 
            unlabeled_category=self.unlabeled_token # Tells model which label to treat as 'unsupervised'
        )
        
        # Train scANVI. 
        # This fine-tunes the embedding using the known labels to separate classes
        # while keeping the unlabeled data aligned via the learned geometry.
        self.model_.train(
            max_epochs=self.max_epochs, 
            accelerator=self.accelerator,
            devices=1 if self.accelerator != "cpu" else "auto"
        )
        
        return self

    def fit_transform(self, x_a, x_b, y_a, y_b):
        self.fit(x_a, x_b, y_a, y_b)
        # Extract the final shared embedding (z)
        return self.model_.get_latent_representation()