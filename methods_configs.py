# Dense / Hiref does not make a difference a1-a4
# UMAP is bad compared to PHATE on a1-a4



methods_params_dict = {
                        # "MALI": {"t:": 'auto'
                        # },





                        "ICML-FoSTA t2": {"method_type": "FoSTA_ICML",
                                        "t": 2},
                        # "ICML-FoSTA auto": {"method_type": "FoSTA_ICML",
                        #                 "t": "auto"},

                        


                        
                        "FoSTA t2": {"method_type": "FoSTA",
                                        "t": 2
                                        },


                        
                        # "FoSTA auto": {"method_type": "FoSTA",
                        #                 "t": "auto"},





                       "FoSTA-kerf t2": {"method_type": "FoSTA",
                                            "kernel_method": "kerf",
                                            "t": 2},




                                            
                        # "FoSTA-kerf auto": {"method_type": "FoSTA",
                        #                     "kernel_method": "kerf",
                        #                     "t": "auto"},

                                            





                      
                      # "New FoSTA t2": {"method_type": "FoSTA",
                      #                      "kernel_method": "gap",
                      #                      "old_version": False,
                      #                      "t": 2,
                      #                      'n_neighbors': 10,
                      #                      "decay": 10,
                      #                      'knn_dist': 'euclidean',
                      #                      },



                        # "New FoSTA t2": {"method_type": "FoSTA",
                        #                      "kernel_method": "gap",
                        #                      "old_version": False,
                        #                      "t": 2,
                        #                      'n_neighbors': 10,
                        #                      "decay": 10,
                        #                      'knn_dist': 'euclidean'
                        #                      },
                        # "New FoSTAker t2": {"method_type": "FoSTA",
                        #                      "kernel_method": "kerf",
                        #                      "old_version": False,
                        #                      "t": 2,
                        #                      'n_neighbors': 10,
                        #                      "decay": 10,
                        #                      'knn_dist': 'euclidean'
                        #                      },
                        
                        # "New FoSTA auto": {"method_type": "FoSTA",
                        #                      "kernel_method": "gap",
                        #                      "old_version": False,
                        #                      "t": 'auto',
                        #                      'n_neighbors': 10,
                        #                      "decay": 10,
                        #                      'knn_dist': 'euclidean'
                        #                      },
                        # "New FoSTAker auto": {"method_type": "FoSTA",
                        #                      "kernel_method": "kerf",
                        #                      "old_version": False,
                        #                      "t": 'auto',
                        #                      'n_neighbors': 10,
                        #                      "decay": 10,
                        #                      'knn_dist': 'euclidean'
                        #                      },
                                             
                                             
                    

                        


                        # "Scanorama":{},



                        # "LIGER":{},

                        
                        # "Harmony":{},



                        # "scVI":{},
                        # "scANVI":{},
                        # "Pamona":{},
                        # "KEMArbf": {},
                        # "KEMAlin": {}

                    }