import numpy as np
import scipy
import pytensor
import pytensor.tensor as pt


RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


def expand_mask(vector, mask, dtype=np.float32):
    where_mask = np.where(mask)
    image = np.zeros(mask.shape, dtype=dtype)
    for idx, (i, j) in enumerate(zip(*where_mask)):
        image[i, j] = vector[idx]
    return image


def VFA(T1,M):
    mat = scipy.io.loadmat('data/synthetic_data.mat')
    FA = np.array(mat['FA']).squeeze()
    FA = FA * np.pi / 180.
    TR = mat['TR'][0][0]

    E = np.exp(-TR/T1) #256X256 
    a = np.multiply(M,1-E)
    if isinstance(T1,np.ndarray):
        up = np.outer(a,np.sin(FA))
        down = 1-np.outer(E,np.cos(FA))
    else:
        FA = pytensor.shared(FA)
        up = pt.outer(a,np.sin(FA))
        down = 1-pt.outer(E,np.cos(FA))
    y = up /down
    return y.squeeze()


def load_data(rs=8):
    k=int(256/rs)
    mat = scipy.io.loadmat('data/synthetic_data.mat')

    ref = mat['ref']
    T1_gt = ref[::k,::k,1]
    M_gt = ref[::k,::k,0]/10e3

    mask = mat['mask'][::k,::k]
    Y = mat['Y'][::k,::k]/10e3
    

    A1 = scipy.sparse.load_npz(f'D_matrix/A1_rs{rs}.npz').tocsc()
    A2 = scipy.sparse.load_npz(f'D_matrix/A2_rs{rs}.npz').tocsc()
    A3 = scipy.sparse.load_npz(f'D_matrix/A3_rs{rs}.npz').tocsc()

        
    return T1_gt, M_gt, mask, A1, A2,A3, Y




    
def aggregate(X):
    return np.mean(X[:,0:1000],axis=1)
    # return X[:,-1]



def D_tv_2D(rs=256):
    mat = scipy.io.loadmat('data/synthetic_data.mat')
    k=int(256/rs)
    mask = mat['mask'][::k, ::k]

    where_mask = np.where(mask)
    num_pixels = np.sum(mask)

    v2i = dict()
    i2v = dict()
    for idx, (i, j) in enumerate(zip(*where_mask)):
        i2v[(i, j)] = idx
        v2i[idx] = (i,j)
    
    A1 = scipy.sparse.lil_matrix((num_pixels, num_pixels), dtype=np.int8)
    A2 = scipy.sparse.lil_matrix((num_pixels, num_pixels), dtype=np.int8)
    A3 = scipy.sparse.lil_matrix((num_pixels, num_pixels), dtype=np.int8)
    A3.setdiag(1)
    idx_A1 = []
    idx_A2 = []
    for idx in range(num_pixels):
        loc = v2i[idx]
        # Neighbor in vertical direction
        loc_neib_i = (loc[0] + 1, loc[1])
        # Neighbor in horizontal direction
        loc_neib_j = (loc[0], loc[1] + 1)
        if loc_neib_i in i2v:
            idx_neib_i = i2v[loc_neib_i]
            A1[idx, idx] = -1
            A1[idx, idx_neib_i] = 1
            idx_A1.append(idx)
        if loc_neib_j in i2v:
            idx_neib_j = i2v[loc_neib_j]
            A2[idx, idx] = -1
            A2[idx, idx_neib_j] = 1
            idx_A2.append(idx)
    # Convert lists to sets for easier operations
    set_A1 = set(idx_A1)
    set_A2 = set(idx_A2)

    # Common elements in both A1 and A2
    X1 = list(set_A1 & set_A2)  # Intersection

    # Elements in A1 but not in A2
    X2 = list(set_A1 - set_A2)  # Difference (A1 - A2)

    # Elements in A2 but not in A1
    X3 = list(set_A2 - set_A1)  # Difference (A2 - A1)

    # Elements in neither A1 nor A2 (assuming a universal reference set)
    reference_set = set(range(0, num_pixels))  # Example universal set {1,2,3,...,9}
    X4 = list(reference_set - (set_A1 | set_A2))  # Elements in neither
    
    # for idx in X4:
    #     A1[idx,idx] = -1

    # Convert to efficient CSR format
    A1 = A1.tocsr()
    A2 = A2.tocsr()
    A3 = A3.tocsr()

    # Save as sparse matrices
    scipy.sparse.save_npz(f'D_matrix/A1_rs{rs}.npz', A1)
    scipy.sparse.save_npz(f'D_matrix/A2_rs{rs}.npz', A2)
    scipy.sparse.save_npz(f'D_matrix/A3_rs{rs}.npz', A3)
    return A1, A2,A3, mask

    
if __name__ == "__main__":
    D_tv_2D()
