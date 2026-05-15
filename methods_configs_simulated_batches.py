# Dense / Hiref does not make a difference a1-a4
# UMAP is bad compared to PHATE on a1-a4



methods_params_dict = {
                        # "ICML-FoSTA t2": {"method_type": "FoSTA_ICML",
                        #                 "t": 2},

                        # "ICML-FoSTA umap": {"method_type": "FoSTA_ICML",
                        #                 "embedder": "UMAP"},
                        # "ICML-FoSTA auto": {"method_type": "FoSTA_ICML",
                        #                 "t": "auto"},

                        



                        # "FoSTA t2": {"method_type": "FoSTA",
                        #                 "t": 2
                        #                 },


                        
                        # "FoSTA auto": {"method_type": "FoSTA",
                        #                 "t": "auto"},





                      #  "FoSTA-kerf t2": {"method_type": "FoSTA",
                      #                       "kernel_method": "kerf",
                      #                       "t": 2},




                                            
                        # "FoSTA-kerf auto": {"method_type": "FoSTA",
                        #                     "kernel_method": "kerf",
                        #                     "t": "auto"},

                                            





                      # "FoSTA-umap": {"method_type": "FoSTA",
                      #                "embedder": "UMAP"},

                      # "FoSTA-et-t2": {"method_type": "FoSTA",
                      #                "t": 2,
                      #                "model_type": "et"},

                      # "FoSTA-et-auto": {"method_type": "FoSTA",
                      #                "t": "auto",
                      #                "model_type": "et"},




                      "FoSTA no pca": {"method_type": "FoSTA",
                                     "t": 2,
                                     "model_type": "rf",
                                     "embedding_basis" : "X"},  


                      "FoSTA_t2": {"method_type": "FoSTA",
                                     "t": 2,
                                     "model_type": "rf"},
                      

                      "FoSTA_t2_balanced": {"method_type": "FoSTA",
                                     "t": 2,
                                     "model_type": "rf",
                                     "class_weight": 'balanced_subsample'},
                      

                      "FoSTA_t2_et": {"method_type": "FoSTA",
                                     "t": 2,
                                     "kernel_method": "kerf",
                                     "model_type": "et",
                                     "bootstrap": True},
                      
                      "FoSTA_tauto_kerf": {"method_type": "FoSTA",
                                     "t": "auto",
                                     "kernel_method": "kerf"},
                      "FoSTA_tauto_gap": {"method_type": "FoSTA",
                                     "t": "auto",
                                     "kernel_method": "gap"},

                        "Scanorama":{},
                        "LIGER":{},
                        "Harmony":{},
                        "scVI":{},
                        "scANVI":{},
                        "MALI":{},
                        "Pamona":{},
                        "KEMArbf": {},
                        "KEMAlin": {}

                    }