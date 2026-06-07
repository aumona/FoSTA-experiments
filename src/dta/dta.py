import graphtools
from scipy import sparse
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import pdb
import ot
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import NearestNeighbors
import pdb
import sklearn
from utils.utils import kernel2Dist, print_mat_stats
from utils.labels import LabelUtils
import logging, os
from scipy.spatial import distance
import warnings 
import tasklogger
import time
from sklearn import preprocessing
from copy import deepcopy
from sklearn.manifold import SpectralEmbedding
from src.phate import PageRankPHATE
from forestkernel import ForestKernel
import umap


class DTA():
    def __init__(self,
             n_components=2,
             embedder = "spectral",
             rfgap=False,
             n_estimators=1000,
             knn=5,
             decay=40,
             t_dpt=1,   # we set this to 1 to compare diffusion (DPT, distances='DPT') VS no diffusion (distances='none')
             t = 'auto',
             beta = 0.7,
             lamb = 1,
             gamma = 1,
             npca=100,
             knn_dist="euclidean",
             knn_max=None,
             n_jobs=-1,
             random_state=None,
             verbose=0,
             njobs=None,
             distances="DPT",  # set to something else to disable DPT
             diff_op_type = "nonsym",
             normalize_rw = False,
             cross_diffusion = "rows",
             distance_gamma = -1,
             entR = 0,
             u = 0.1,
             r = 3,
             m = 1,
             distance = 'cosine',
             anisotropy = 0,
             constrW = 'kernel',
             **kwargs):
        
        '''
        
        entR: Entropy regularization parameter Optimal Transport 
        
        m: default=1, 0 < m < 1 computes Partial transport 
            Percentage of mass from domain 1 to be transported to the second domain 
            Set to m>1 for auto selection of m (useful for uneven dataset sizes)
        
 
        
        '''
        
        self.n_components = n_components
        self.embedder = embedder
        self.rfgap = rfgap
        self.n_estimators = n_estimators
        self.decay = decay
        self.knn = knn
        self.t_dpt = t_dpt
        self.t = t
        self.beta = beta
        self.npca = npca
        self.knn_dist = knn_dist
        self.knn_max = knn_max
        self.random_state = random_state
        self.kwargs = kwargs

        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.distances = distances
        self.diff_op_type = diff_op_type
        self.normalize_rw = normalize_rw
        self.cross_diffusion = cross_diffusion
        self.lamb = lamb
        self.distance_gamma = distance_gamma
        self.entR = entR
        self.u = u
        self.r = r
        self.distance = distance
        self.m = m
        self.anisotropy = anisotropy
        self.constrW = constrW
        self.normalize_priors = 1  # to match RFMALI
        self.normalize_M = 1
        
    def compute_graphs(self):
        
        
        self.N1, self.f1 = self.domain1.shape
        self.N2, self.f2 = self.domain2.shape
        
        self.Nshared = 0 
        if self.met != 'mali':
            self.domain1 = np.vstack((self.domain1, self.sharedD1))
            self.domain2 = np.vstack((self.domain2, self.sharedD2))
            self.Nshared = self.sharedD2.shape[0]
            if self.met == 'dta':
                self.rfgap = False
        

        
        if not self.rfgap:
            n_pca1 = np.minimum(self.npca, self.f1)
            if n_pca1 < 100:
                n_pca1 = None

            self.graphD1 = graphtools.Graph(self.domain1,
                                            n_pca=n_pca1,
                                            distance=self.knn_dist,
                                            knn=self.knn,
                                            decay=self.decay,
                                            thresh=1e-4,
                                            n_jobs=self.n_jobs,
                                            verbose=self.verbose,
                                            random_state=self.random_state,
                                            **(self.kwargs))
            

            # pdb.set_trace()
            n_pca2 = np.minimum(self.npca, self.f2)
            if n_pca2 < 100:
                n_pca2 = None
            self.graphD2 = graphtools.Graph(self.domain2,
                                            n_pca=n_pca2,
                                            distance=self.knn_dist,
                                            knn=self.knn,
                                            decay=self.decay,
                                            thresh=1e-4,
                                            n_jobs=self.n_jobs,
                                            verbose=self.verbose,
                                            random_state=self.random_state,
                                            **(self.kwargs))
        else:
            rfgap_params = {
                'random_state': self.random_state,
                'prediction_type': 'classification',  # force classification mode
                'n_estimators': self.n_estimators,
                'kernel_method': 'gap',
                'model_type': 'rf',
                'force_nonzero_diag': True,
                'verbose': 0,
                'n_jobs': self.n_jobs,
            }
            rfgap = ForestKernel(**rfgap_params)
            mask_unlabeled_a = LabelUtils.get_unlabeled_mask(self.labels1)
            idx_unlabeled_a = np.flatnonzero(mask_unlabeled_a)
            rfgap.fit(self.domain1, self.labels1, idx_unlabeled=idx_unlabeled_a)
            prox1 = rfgap.get_kernel(normalize_diagonal=True)
            self.graphD1 = graphtools.Graph(prox1,
                                            precomputed='affinity',
                                            n_jobs=self.n_jobs,
                                            kernel_symm=None,
                                            verbose=self.verbose,
                                            random_state=self.random_state,
                                            **(self.kwargs))
            mask_unlabeled_b = LabelUtils.get_unlabeled_mask(self.labels2)
            idx_unlabeled_b = np.flatnonzero(mask_unlabeled_b)
            rfgap.fit(self.domain2, self.labels2, idx_unlabeled=idx_unlabeled_b)
            prox2 = rfgap.get_kernel(normalize_diagonal=True)
            self.graphD2 = graphtools.Graph(prox2,
                                            precomputed='affinity',
                                            n_jobs=self.n_jobs,
                                            kernel_symm=None,
                                            verbose=self.verbose,
                                            random_state=self.random_state,
                                            **(self.kwargs))
        
        '''Diffusion operators'''
        self.p1 = self.graphD1.diff_op
        self.p2 = self.graphD2.diff_op

        
        '''Diffuse t steps'''    
        self.p1_t = np.linalg.matrix_power(self.p1.toarray(), self.t_dpt)
        self.p2_t = np.linalg.matrix_power(self.p2.toarray(), self.t_dpt)
        
    
    def compute_labels_distance(self):
        # combine labels if shared labels are provided
        if (self.labelsh1 is not None) and (self.labelsh2 is not None):
            self.labels1 = np.concatenate((self.labels1, self.labelsh1))
            self.labels2 = np.concatenate((self.labels2, self.labelsh2))
    
        labels1 = np.asarray(self.labels1)
        labels2 = np.asarray(self.labels2)
    
        # keep only valid labels (exclude masked / unlabeled)
        valid1 = ~np.isnan(labels1) if np.issubdtype(labels1.dtype, np.floating) else np.ones(len(labels1), dtype=bool)
        valid2 = ~np.isnan(labels2) if np.issubdtype(labels2.dtype, np.floating) else np.ones(len(labels2), dtype=bool)
    
        valid1 &= (labels1 != -1)
        valid2 &= (labels2 != -1)
    
        labels1_valid = labels1[valid1]
        labels2_valid = labels2[valid2]
    
        self.unique_labels = np.intersect1d(labels1_valid, labels2_valid)
    
        self.gamma1_c = np.zeros((self.domain1.shape[0], len(self.unique_labels)))
        self.gamma2_c = np.zeros((self.domain2.shape[0], len(self.unique_labels)))
    
        # priors as dictionaries aligned with actual label values
        vals1, cnts1 = np.unique(labels1_valid, return_counts=True)
        vals2, cnts2 = np.unique(labels2_valid, return_counts=True)
    
        priors1 = {lab: cnt / len(labels1_valid) for lab, cnt in zip(vals1, cnts1)}
        priors2 = {lab: cnt / len(labels2_valid) for lab, cnt in zip(vals2, cnts2)}
    
        eps = 1e-12
    
        for p, lab in enumerate(self.unique_labels):
            indx1 = np.where(labels1 == lab)[0]
            indx2 = np.where(labels2 == lab)[0]
    
            if len(indx1) == 0 or len(indx2) == 0:
                continue
    
            if self.distances == "DPT":
                base1 = self.M1
                base2 = self.M2
            else:
                base1 = self.p1_t
                base2 = self.p2_t
    
            if self.normalize_priors == 1:
                denom1 = max(priors1.get(lab, 0.0), eps)
                denom2 = max(priors2.get(lab, 0.0), eps)
    
                self.gamma1_c[:, p] = np.sum(base1[:, indx1], axis=1) / denom1
                self.gamma2_c[:, p] = np.sum(base2[:, indx2], axis=1) / denom2
            else:
                self.gamma1_c[:, p] = np.sum(base1[:, indx1], axis=1) / max(len(indx1), 1)
                self.gamma2_c[:, p] = np.sum(base2[:, indx2], axis=1) / max(len(indx2), 1)
    
        # sanitize before cdist
        self.gamma1_c = np.nan_to_num(self.gamma1_c, nan=0.0, posinf=1e6, neginf=-1e6)
        self.gamma2_c = np.nan_to_num(self.gamma2_c, nan=0.0, posinf=1e6, neginf=-1e6)
    
        # avoid cosine NaNs from zero rows
        row_norms1 = np.linalg.norm(self.gamma1_c, axis=1)
        row_norms2 = np.linalg.norm(self.gamma2_c, axis=1)
    
        zero1 = row_norms1 < eps
        zero2 = row_norms2 < eps
    
        if self.distance == "cosine":
            self.gamma1_c[zero1, 0] = eps
            self.gamma2_c[zero2, 0] = eps
    
        self.DistancesLabels = cdist(self.gamma1_c, self.gamma2_c, self.distance)
        self.DistancesLabels = np.nan_to_num(self.DistancesLabels, nan=1.0, posinf=1.0, neginf=1.0)

    
    def compute_corres_distance(self): 
        
            # compute distance using correspondences 
        
        if self.distances == "DPT":
            print("using DPT")
            gamma1_c = self.M1[:, -self.Nshared:]
            gamma2_c = self.M2[:, -self.Nshared:]
            
        else: 
            print("Not using DPT")
            gamma1_c = self.p1_t[:, -self.Nshared:]
            gamma2_c = self.p2_t[:, -self.Nshared:]
         
        self.DistancesCorres = cdist(gamma1_c, gamma2_c, self.distance)
        self.DistancesCorres[np.isnan(self.DistancesCorres)] = 1
            
        

    def compute_diffusion_distances(self):

        self.Distances12 = None
        
        if self.distances == 'DPT':
            self.compute_dpt()
            
        if self.met == "both":  
        
           self.compute_labels_distance()
           self.compute_corres_distance() 
            
           self.Distances12 = self.DistancesCorres + self.DistancesLabels 
        
        elif self.met == "mali":
           self.compute_labels_distance()
           self.Distances12 = self.DistancesLabels 
        
        elif self.met == "dta":
           self.compute_corres_distance()
           self.Distances12 = self.DistancesCorres 

            

    def compute_dpt(self, ridge=1e-8):
        rng = np.random.default_rng(self.random_state)

        # Domain 1
        v0 = rng.random(self.p1.shape[0])
        w, rv = scipy.sparse.linalg.eigs(self.p1, k=1, v0=v0)
        w, lv = scipy.sparse.linalg.eigs(self.p1.transpose(), k=1, v0=v0)
    
        P = self.p1.toarray()
        A1 = np.eye(P.shape[0]) - (P - np.outer(rv.real, lv.real))
        A1 = A1 + ridge * np.eye(A1.shape[0])
        self.M1 = np.linalg.solve(A1, np.eye(A1.shape[0])) - np.eye(A1.shape[0])
    
        # Domain 2
        v0 = rng.random(self.p2.shape[0])
        w, rv = scipy.sparse.linalg.eigs(self.p2, k=1, v0=v0)
        w, lv = scipy.sparse.linalg.eigs(self.p2.transpose(), k=1, v0=v0)
    
        P = self.p2.toarray()
        A2 = np.eye(P.shape[0]) - (P - np.outer(rv.real, lv.real))
        A2 = A2 + ridge * np.eye(A2.shape[0])
        self.M2 = np.linalg.solve(A2, np.eye(A2.shape[0])) - np.eye(A2.shape[0])
    
        if self.normalize_M == 1:
            min_max_scaler = preprocessing.MinMaxScaler()
            self.M1 = min_max_scaler.fit_transform(self.M1.transpose()).transpose()
            self.M2 = min_max_scaler.fit_transform(self.M2.transpose()).transpose()

   
    def optimal_transport(self):


 
        if self.N1 == self.N2:
            if self.m == 1:
                self.a = np.repeat(1., self.N1)
                self.b = np.repeat(1., self.N1)
                if self.entR == 0:
                    # Compute OT 
                    self.transport = "wot"
                else:
                    self.transport = "wotR"

            else:
                if self.entR == 0:
                    # Compute OT 
                    self.transport = "wotpartial"
                    self.a = np.repeat(1/self.N1, self.N1)
                    self.b = np.repeat(1/self.N1, self.N1)
                    self.m = np.floor(self.m*self.N1)/self.N1
                else:
                    self.transport = "wotpartialR"
                    self.m = np.floor(self.m*self.N1)/self.N1
                    self.a = np.repeat(1/self.N1, self.N1)
                    self.b = np.repeat(1/self.N1, self.N1)
        else:
            if self.m != 1:
                self.a = np.repeat(1/self.N1, self.N1)
                self.b = np.repeat(1/self.N2, self.N2)
                self.m = np.floor(self.m*self.N1)/self.N1
                if self.entR > 0:
                    self.transport = "wotpartialR"
                else:
                    self.transport = "wotpartial"
            
            elif self.m == 1:
                self.transport = "wot"
                if self.entR > 0:
                    self.transport = "wotR"
                self.a = np.repeat(1, self.N1).astype(float)
                self.b = np.repeat(self.N1/self.N2, self.N2)
                print("Unbalanced")
        

        
        a = self.a
        b = self.b

        if self.transport == "wot":
            self.T = ot.emd(a, b, self.Distances12[:self.N1, :self.N2])
        elif self.transport == "wotR":
            # self.T = ot.sinkhorn(a,b, self.Distances12[:self.N1, :self.N2], reg = self.entR, 
            #                       numItermax = 10000)
            
            
            # self.T = ot.bregman.greenkhorn(a,b, self.Distances12[:self.N1, :self.N2], reg = self.entR,
            #                                              numItermax = 100000)
            
            # self.T = ot.bregman.sinkhorn_epsilon_scaling(a,b, self.Distances12[:self.N1, :self.N2], reg = self.entR,
            #                                              numItermax = 1000)
            # self.T = ot.bregman.screenkhorn(a,b, self.Distances12[:self.N1, :self.N2], reg = self.entR)
            
            self.T = ot.bregman.sinkhorn_log(a,b, self.Distances12[:self.N1, :self.N2], reg = self.entR)
            # print(f"entropic {self.entR}")
        elif self.transport == "wotpartial":
            # self.Distances12[:self.N1, :self.N2] = self.Distances12[:self.N1, :self.N2] * np.random.beta(1, 0.1, size= (self.N1, self.N2))
            self.T = ot.partial.partial_wasserstein(a, b, self.Distances12[:self.N1, :self.N2], m = self.m, 
                                                    nb_dummies = 100)
            self.T[self.T < 1e-10] = 0
            
        elif self.transport == "wotpartialR":
            import ipdb
            ipdb.set_trace()
            
            self.T = ot.partial.entropic_partial_wasserstein(a, b, self.Distances12[:self.N1, :self.N2], reg = self.entR, m = self.m)
            self.T[self.T < 1e-10] = 0

        else:
            raise ValueError("Not implemented")
            
        self.T[self.T < 1e-5] = 0


    def fit(self, domain1, domain2, sharedD1 = None, sharedD2 = None, 
            labels1 = None, labels2 = None, labelsh1 = None, labelsh2 = None):
        
        self.domain1 = domain1
        self.domain2 = domain2
        self.sharedD1 = sharedD1
        self.sharedD2 = sharedD2
        self.labels1 = labels1
        self.labels2 = labels2
        self.labelsh1 = labelsh1
        self.labelsh2 = labelsh2
        
        if (self.sharedD1 is None) or (self.sharedD2 is None):
            if (self.labels1 is None) or (self.labels2 is None):
                raise ValueError("No shared or label information between domains")
            else:
                print("No known correspondences, only using label information")
                self.met = "mali"
        else:
            if (self.labels1 is None) or (self.labels2 is None):
                print("Only using known correspondences, not using label information")
                self.met = "dta"
            else: 
                print("Using both known correspondences and label information")
                self.met = "both"
            
            
        print("computing graphs")
        start_time = time.time()
        self.compute_graphs()
        end_time = time.time()
        self.graphtime = end_time - start_time
        if self.verbose > 0:
            print(f" computing graphs took: {self.graphtime}")
        
        start_time = time.time()
        self.compute_diffusion_distances()
        end_time = time.time()
        self.difftime = end_time - start_time
        if self.verbose > 0:
            print(f" computing diffusion distances took: {self.difftime}")
        
        optimal_m = None
        costs = None
        tried_ms = None
        
        if self.m > 1 :
            print("finding the best m")
            optimal_m, costs, tried_ms = self.find_optimal_mass()


        print("computing OT")
        start_time = time.time()
        self.optimal_transport()
        end_time = time.time()
        self.ottime = end_time - start_time
        if self.verbose > 0:
            print(f" computing OT took: {self.ottime}")
       

        self.store_results()

        return self.T, optimal_m, costs, tried_ms
    
 
    
    def store_results(self):

        T = self.T/self.T.max()
        indx1, indx2 = np.nonzero(self.T)
        
        Tc = np.zeros((self.N1+self.Nshared, self.N2+self.Nshared))
        Tc[:self.N1, :self.N2] = T
        Tc[self.N1:, self.N2:] = np.eye((self.Nshared))
        
        
        if self.constrW == 'diffusion':
            W1 = self.graphD1.K.toarray()
            W2 = self.graphD2.K.toarray()
        elif self.constrW == 'kernel':
            W1 = self.graphD1.K.toarray().astype(float)
            W2 = self.graphD2.K.toarray().astype(float)
        
            d1 = np.diag(W1).copy()
            d2 = np.diag(W2).copy()
        
            eps = 1e-12
            d1[d1 <= eps] = 1.0
            d2[d2 <= eps] = 1.0
        
            W1 = W1 / d1[:, None]
            W2 = W2 / d2[:, None]
        
            W1 = np.nan_to_num(W1, nan=0.0, posinf=0.0, neginf=0.0)
            W2 = np.nan_to_num(W2, nan=0.0, posinf=0.0, neginf=0.0)      
        W12 = np.dot(W1, Tc)
        W21 = np.dot(W2, Tc.transpose())
        self.W = np.block([[W1, W12], [W21, W2]])
        self.W = self.W + self.W.transpose() 
        
        self.W2 = np.block([[W1, Tc], [Tc.transpose(), W2]]) 
            # self.W[np.abs(self.W) < 1e-4] = 0
            
        # W12 = np.zeros(self.T.shape)
        # W12[np.ix_(indx1,)] = W2[np.ix_(indx2,)]
        
        #concatenation of features
        # data_c = np.hstack((self.domain1, self.domain2))
        # data_c[indx1, self.domain1.shape[1]:(self.domain1.shape[1]+self.domain2.shape[1])] = self.domain2[np.ix_(indx2,)]
        # self.data_c = data_c
        # self.ind_cost = (self.Distances*self.T).sum(axis = 1)
        self.cost = np.sum(self.Distances12[:self.N1, :self.N2] * self.T[:self.N1, :self.N2])/self.m

        
        if self.verbose:
            T = Tc
            print_mat_stats("COUPLING MATRIX:", T)
            print_mat_stats("Within-domain A (W1)", W1)
            print_mat_stats("Within-domain B (W2)", W2)
            print_mat_stats("Cross-domain A→B (W12)", W12)
            print_mat_stats("Cross-domain B→A (W21)", W21)


    def embed(self):
        
        if self.embedder == "spectral" :
            embedding = SpectralEmbedding(
            n_components=self.n_components, affinity='precomputed', random_state=self.random_state)

            embedding_joint = embedding.fit_transform(self.W)

        elif self.embedder =="UMAP":
            #embedding = UMAP(knn_dist='precomputed_affinity')

            DistM = kernel2Dist(self.W)

            embedding_joint = umap.UMAP(n_components=self.n_components,
                metric='precomputed', random_state=self.random_state).fit_transform(DistM)

        elif self.embedder =="PHATE":

            embedding = PageRankPHATE(n_components=self.n_components, knn_dist='precomputed_affinity',
                                      beta=self.beta, t=self.t, random_state=self.random_state)
            embedding_joint = embedding.fit_transform(self.W)

        elif self.embedder == "barycentric":
#             import ipdb
#             ipdb.set_trace()

#             normalizer = sklearn.preprocessing.MinMaxScaler()
#             T_tilde = normalizer.fit_transform(self.T.transpose()).transpose()

            normalizer = sklearn.preprocessing.MinMaxScaler()
            
        
#             T_tilde = self.T
        
            if self.N1 > self.N2:
                T_tilde_transpose = normalizer.fit_transform(self.T.transpose())
                embedding_y = np.dot(T_tilde_transpose, self.domain1)
                embedding_joint = np.concatenate((self.domain1, embedding_y))
            else:
                T_tilde = normalizer.fit_transform(self.T)
                embedding_x = np.dot(T_tilde, self.domain2)
                embedding_joint = np.concatenate(( embedding_x, self.domain2))
        
            self.embedding = embedding_joint
        return embedding_joint
    
    def find_optimal_mass_linspace(self):
        models = {}
        ms = np.linspace(0, 1, num = 26)
        ms = ms[1:len(ms)-1]
        
        for m in ms:
            print("trying partial OT with m = {}".format(m) )
            models[m] = deepcopy(self)
            models[m].m = m

            models[m].optimal_transport() # this function modifies self so that's why we made copies above
            models[m].store_results()

        costs = np.asarray( [models[m].cost for m in ms])

        # find elbow point
        slope = np.insert(costs, 0,0) - np.append(costs, 0)
        slope = slope[1:len(slope)-1]

        change_in_slope = np.insert(slope, 0,0) - np.append(slope, 0)
        index_optimal_m = np.argmax(change_in_slope)
        
        optimal_m = ms[index_optimal_m]
        print("Optimal m was selected to be m = {}".format(optimal_m))
        self.m = optimal_m
        

        return optimal_m, costs, ms.tolist()
    
    
    def find_optimal_mass(self):
        models = {}

        num_searches = 0
        max_num_searches = 20
        precision = 0.005
        ms = [0.001, 0.333, 0.666, 0.999]
        tried_ms = [] 
        
        while len(ms) > 0:

            m = ms.pop()
            num_searches += 1

            print("trying partial OT with m = {}".format(m) )
            models[m] = deepcopy(self)
            models[m].m = m

            models[m].optimal_transport() # this function modifies self so that's why we made copies above
            models[m].store_results()

            
            tried_ms.append(m)
            tried_ms.sort()

            if len(ms) == 0: 
                if num_searches > max_num_searches - 2:
                    break

                # add some values to try
                costs = np.asarray( [models[m].cost for m in tried_ms])
                diff_costs = np.insert(costs, 0,0) - np.append(costs, 0)
                diff_ms = np.insert(tried_ms, 0,0) - np.append(tried_ms, 0)
                
                diff_costs = diff_costs[1:len(diff_costs)-1]
                diff_ms = diff_ms[1:len(diff_ms)-1]
                slope = diff_costs / diff_ms

                #slope = slope[1:len(slope)-1]
                diff_ms2 = np.insert(tried_ms, [0,0], 0) - np.append(tried_ms, [0,0])
                diff_ms2 = diff_ms2[2:len(diff_ms2)-2]

                change_slope = np.insert(slope, 0,0) - np.append(slope, 0)
                change_slope = change_slope[1:len(change_slope)-1]
                change_slope = change_slope/diff_ms2
                
                #change_change_slope = np.insert(change_slope, 0,0) - np.append(change_slope, 0)
                
                # if the second best interval is very close in cost to the best one, also add it
                
#                 import ipdb
#                 ipdb.set_trace()
                
                index_best_interval = np.argsort(change_slope, axis=0)[-1]
                index_intervals_to_add = [index_best_interval]
                index_second_best_interval = np.argsort(change_slope, axis=0)[-2]
                if np.abs(change_slope[index_best_interval] - change_slope[index_second_best_interval])/ np.max((change_slope[index_best_interval], change_slope[index_second_best_interval])) < 0.3:
                    index_intervals_to_add.append(index_second_best_interval)
                
                
                for index_interval in index_intervals_to_add :
                    interval_start = tried_ms[index_interval]
                    interval_end = tried_ms[index_interval + 2]
                
                    if interval_end - interval_start < precision:
                        break

                
                    new_m1 = interval_start + (interval_end - interval_start)/3
                    new_m2 = interval_end - (interval_end - interval_start)/3
                    new_m1 = np.floor(new_m1*1000)/1000
                    new_m2 = np.floor(new_m2*1000)/1000
                    ms = ms + [new_m1, new_m2]
                

        # plot m against cost

        costs = np.asarray( [models[m].cost for m in tried_ms])
        #plt.scatter(tried_ms, costs)
        #plt.savefig("m_parameter_search.png")
        
        
        # find elbow point
        diff_costs = np.insert(costs, 0,0) - np.append(costs, 0)
        diff_ms = np.insert(tried_ms, 0,0) - np.append(tried_ms, 0)
        slope = diff_costs / diff_ms
        slope = slope[1:len(slope)-1]
        diff_ms2 = np.insert(tried_ms, [0,0], 0) - np.append(tried_ms, [0,0])
        diff_ms2 = diff_ms2[2:len(diff_ms2)-2]

        change_slope = np.insert(slope, 0,0) - np.append(slope, 0)
        change_slope = change_slope[1:len(change_slope)-1]
        change_slope = change_slope/diff_ms2
        
        index_optimal_m = np.argmax(change_slope)
        
        optimal_m = tried_ms[index_optimal_m]
        print("Optimal m was selected to be m = {}".format(optimal_m))
        self.m = optimal_m
        # change parameters of self to equal the best model

        return optimal_m, costs, tried_ms
