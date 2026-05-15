# methods_params_dict = {"ICML FoSTA t=2": {"method_type": "FoSTA_ICML",
#                                         "t": 2}
#                         # "Harmony": {}
#                     }



# methods_params_dict = {"FoSTA Spectral mu=0.2": {"method_type": "FoSTA_ICML",
#                                       "embedder": "spectral",
#                                        "mu": 0.2},
#                         "FoSTA Spectral mu=0.5": {"method_type": "FoSTA_ICML",
#                                                               "embedder": "spectral",
#                                                                "mu": 0.5},
#                         "FoSTA Spectral mu=0.8": {"method_type": "FoSTA_ICML",
#                                                               "embedder": "spectral",
#                                                                "mu": 0.8}                  
#                     }


# methods_params_dict = {"FoSTA Spectral mu=0.95": {"method_type": "FoSTA_ICML",
#                                       "embedder": "spectral",
#                                         "mu": 0.95}, 
#                        "FoSTA Spectral mu=1": {"method_type": "FoSTA_ICML",
#                                       "embedder": "spectral",
#                                         "mu": 1.0},              
#                     }


methods_params_dict = { "FoSTA no pca": {"method_type": "FoSTA",
                                     "t": 2,
                                     "model_type": "rf",
                                     "embedding_basis" : "X"},          
                    }