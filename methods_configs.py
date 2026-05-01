methods_params_dict = {
    
                    #     "Old FoSTA kerf t=2": {"method_type": "FoSTA",
                    #                         "kernel_method": "kerf",
                    #                         "old_version": True,
                    #                         "t": 2},
                    #    "Old FoSTA kerf t=auto": {"method_type": "FoSTA",
                    #                         "kernel_method": "kerf",
                    #                         "old_version": True,
                    #                         "t": "auto"},


                        "MALI": {"t:": 'auto'
                        },





                        "ICML FoSTA t=2": {"method_type": "FoSTA_ICML",
                                        "t": 2},
                        "ICML FoSTA t=auto": {"method_type": "FoSTA_ICML",
                                        "t": "auto"},







                        
                        "Old FoSTA gap t=2": {"method_type": "FoSTA",
                                        "kernel_method": "gap",
                                        "old_version": True,
                                        "t": 2},
                        
                        "Old FoSTA gap t=2 (max-normalized rows)": {"method_type": "FoSTA",
                                        "kernel_method": "gap",
                                        "max_normalize": True,
                                        "old_version": True,
                                        "t": 2},

                       "Old FoSTA gap t=auto": {"method_type": "FoSTA",
                                            "kernel_method": "gap",
                                            "old_version": True,
                                            "t": "auto"},
                        "Old FoSTA gap t=auto (max-normalized rows)": {"method_type": "FoSTA",
                                            "kernel_method": "gap",
                                            "max_normalize": True,
                                            "old_version": True,
                                            "t": "auto"},

                                            

                        # "New FoSTA kerf t=2": {"method_type": "FoSTA",
                        #                     "kernel_method": "kerf",
                        #                     "old_version": False,
                        #                     "t": 2},
                        # "New FoSTA kerf t=auto": {"method_type": "FoSTA",
                        #                     "kernel_method": "kerf",
                        #                     "old_version": False,
                        #                     "t": "auto"},



                        "New FoSTA gap t=2": {"method_type": "FoSTA",
                                             "kernel_method": "gap",
                                             "old_version": False,
                                             "t": 2},
                                             
                        "New FoSTA gap t=auto": {"method_type": "FoSTA",
                                             "kernel_method": "gap",
                                             "old_version": False,
                                             "t": "auto"},
                        
                        "New FoSTA kerf t=2": {"method_type": "FoSTA",
                                             "kernel_method": "kerf",
                                             "old_version": False,
                                             "t": 2},
                                             
                        "New FoSTA kerf t=auto": {"method_type": "FoSTA",
                                             "kernel_method": "kerf",
                                             "old_version": False,
                                             "t": "auto"},

                        


                        # "Scanorama":{},
                        # "LIGER":{},
                        # "Harmony":{},
                        # "scVI":{},
                        # "scANVI":{},
                        # "Pamona":{},
                        # "KEMArbf": {},
                        # "KEMAlin": {}

                    }