import numpy as np


# example of use :
# for binary trees with a merged branch in the second domain (two branches in the same direction, indistinguishable)
# tree1, branches1 = gen_tree(n_branch=5, n_child=2, branch_length=100, sigma=0, merged_branch = False)
# tree2, branches2 = gen_tree(n_branch=5, n_child=2, branch_length=100, sigma=0, merged_branch = True)

# for two different trees, no merged branch:
# tree1, branches1 = gen_tree(n_branch=5, branch_length=100, sigma=0, merged_branch = False)
# tree2, branches2 = gen_tree(n_branch=5, branch_length=100, sigma=0, merged_branch = False)

# can play with the structure of the tree by changing n_child to a list, e.g. n_child = [3,2,2] for a tree with 3 branches at the root, then 2 branches at each subsequent split


# then visualize with:
# viz_tree(tree1, branches1, methods=["PCA", "PHATE", "UMAP"], title= "Tree 1")


def gen_dla(n_dim=100, n_branch=20, branch_length=100, rand_multiplier=2, seed=37, sigma=4, merged_branch = True):
    # method taken from phate (phate.tree.gen_dla)
    # modifed to add option for merged branch
    # in my experiments, it's hard to control the noise and the results are not so clean. see gen_tree below for a cleaner tree structure.
    
    if merged_branch:
        n_branch = n_branch - 1 # because the last one will be double
        print(n_branch)
    
    np.random.seed(seed)
    M = np.cumsum(rand_multiplier *  ( -1 + 2*np.random.rand(branch_length, n_dim))  , 0)    
    C_last_branch=np.array([])
    
    for i in range(n_branch):
        rng = np.random.default_rng()
        ind = rng.integers(branch_length)
        
        # new branch direction
        -1 + 2* np.random.rand(1, n_dim)
        
        if (not merged_branch) or i < n_branch - 1 :
            A = rand_multiplier * (-1 + 2*np.random.rand(branch_length, n_dim))
           
        else: # if merged branch, at the last branch put twice as many points
            print("merging last branch")
            
            A = rand_multiplier * (-1 + 2*np.random.rand(2*branch_length, n_dim))
            # randomly assign labels within this last branch. with equal numbers
            C_last_branch = np.concatenate((n_branch * np.ones(branch_length), (n_branch+1) * np.ones(branch_length)))
            C_last_branch = rng.permutation(C_last_branch)

        
        new_branch = np.cumsum(A, 0)
        M = np.concatenate([M, new_branch + M[ind, :]])
        
    noise = np.random.normal(0, sigma, M.shape)
    M = M + noise

    
    # returns the group labels for each point to make it easier to visualize
    # embeddings
    C = np.array([i // branch_length for i in range(n_branch * branch_length)])   
    C = np.concatenate((C, C_last_branch))
       
    return M, C



# pick a random direction, place n points in that direction

def gen_tree(n_branch=20, n_child = 2, n_dim_per_branch = 4, branch_length=100, seed=37, sigma=4, merged_branch = False) : 
    ''' function to generate tree-like dataset with branches in independent directions.
    Each branch is in its own set of dimensions, so the branches are independent.
    
    inputs:
    n_dim_per_branch : number of independent dimensions per branch
    n_branch : number of branches
    n_child : number of branches coming out of each branch. 
        should either be an int, or a list with an int for each desired split. the sum of the list should equal the number of branches.
        if int, assume the tree is regular
    branch_length : number of points per branch
    seed : random seed
    sigma : noise level
    merged_branch : if True, the last two branches will be in the same direction (i.e. merged)
    
    outputs:
    T : data matrix (n_branch * branch_length) x (n_dim_per_branch * n_branch)
    C : branch labels for each point (length n_branch * branch_length)
    root: idx of the root
    '''
    
    # here inside the function, i set the length of the branches. this affects the relative impact of sigma. this value may be too small
    norm = 100 # length i.e. euclidean norm of the branches 
    end_point = np.zeros((1,n_dim_per_branch * n_branch))
    
    T = None
    cur_child = 0
    cur_split = 0
    endpoints = []  # stack/queue of endpoints to use for branching
    
    for i in range(n_branch):
       
        if not merged_branch or i < n_branch-1 :  
            direction = np.random.rand(n_dim_per_branch) # random vector indictating a direction
            direction = direction/np.sum(direction**2) # make it a unit vector
        else:  # at the last branch, set the direction to be the same as the previous one. i.e. do not update the direction        
            print("last branch, keeping previous direction")
            # set i as i-1 so that twe use the same coordinates
            i = i-1
        
        points_along_line = np.random.rand(branch_length) * norm # could also do a random uniform sampling along that direction
        
        new_branch = np.zeros((branch_length, n_dim_per_branch * n_branch)) # points in branch x dims
        new_branch_few_dim = np.outer(points_along_line, direction) # in only n_dim_per_branch dimensions
        
        # pad with zeros and add the previous endpoint
        new_branch[:, i*n_dim_per_branch: (i+1)*n_dim_per_branch ] =  new_branch_few_dim
        new_branch = new_branch + end_point
        

        if T is None:
            T = new_branch
            
            idx_root = np.argmin(points_along_line)
            
            
            idx_max = np.argmax(points_along_line)
            end_point = new_branch[idx_max,:] # set the last point of the first branch as endpoint.
        else:
            T = np.concatenate([T, new_branch])   
            
            # add current endpoint of branch to the stack/queue
            idx_max = np.argmax(points_along_line)
            new_end_point = new_branch[idx_max,:] # set the last point of the branch as endpoint.
            endpoints.append(new_end_point)
            
            # check if we need to update the endpoint based on the number of children
            cur_n_child = n_child if isinstance(n_child, int) else n_child[cur_split]
            if cur_child >= cur_n_child:
                # update the endpoint to be the last added one
                end_point = endpoints.pop(0) # pop the oldest endpoint added                
                # reset the child counter
                cur_child = 0
                cur_split = cur_split + 1
        
        cur_child = cur_child + 1
        
        # i should have a stack or queue of endpoints to choose from to make more complex trees.
            
    noise = np.random.normal(0, sigma, T.shape)
    T = T + noise

        
    # returns the branch labels for each point to make it easier to visualize
    # embeddings
    C = np.array([i // branch_length for i in range(n_branch * branch_length)])

    return T, C, idx_root
