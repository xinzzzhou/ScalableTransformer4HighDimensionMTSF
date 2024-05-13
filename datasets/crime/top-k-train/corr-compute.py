import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import deepgraph as dg


# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    features_s = df_x[index_s]
    features_t = df_x[index_t]
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples_
    return corr

# parallel computation
def create_ei(i):

    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)

    # store edge table
    g.e.to_pickle(f'tmp/correlations-train-{data}/{str(i).zfill(3)}.pickle')

# computation
if __name__ == '__main__':
    root_path = './datasets/'
    df_dataset = 'crime.csv'
    data='crime'

    df = pd.read_csv(root_path+df_dataset, header = None)
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True, drop=True)

    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    df_train = df.iloc[:num_train]

    df_x = df_train.values
    df_x = df_x.astype(float)
    df_x = df_x.T
    df_x = (df_x - df_x.mean(axis=1, keepdims=True)) / df_x.std(axis=1, keepdims=True)
    np.save(f'samples_train_{data}', df_x)

    n_features_ = int(1155)
    n_samples_ = int(182)
    step_size = 1e6
    n_processes = 1000
    # load samples as memory-map
    df_x = np.load(f'samples_train_{data}.npy', mmap_mode='r')
    # create node table that stores references to the mem-mapped samples
    v = pd.DataFrame({'index': range(df_x.shape[0])})
    # index array for parallelization
    pos_array = np.array(np.linspace(0, n_features_*(n_features_-1)//2, n_processes), dtype=int)

    
    # computation
    os.makedirs(f'tmp/correlations-train-{data}', exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass
    
    
    # store correlation values
    files = os.listdir(f'tmp/correlations-train-{data}/')
    files.sort()
    store = pd.HDFStore(f'e-train-{data}.h5', mode='w')
    for f in files:
        et = pd.read_pickle(f'tmp/correlations-train-{data}/{f}')
        store.append('e', et, format='t', data_columns=True, index=False)
    store.close()
    
    # transfer correlation table --> matrix
    e = pd.read_hdf(f'e-train-{data}.h5')
    max_index = max(e.index.get_level_values(1)) + 1
    matrix = np.full((max_index, max_index), np.nan)
    for (s, t), value in e.iterrows():
        matrix[s, t] = value
        matrix[t, s] = value
        
    # get the sorted indices, fill nan with 0
    matrix = np.nan_to_num(matrix)
    # for each row of matrix, rank the index according to the value (descending), 
    matrix_rank = np.fliplr(matrix.argsort(axis=1))
    tr = matrix_rank
    # for each row of tr, remove the value that the same as the index
    results=[]
    for i in range(1155):
        tmp = []
        for j in range(1155):
            if tr[i][j] != i:
                tmp.append(tr[i][j])
        results.append(tmp)
        
    result_arr = np.array(results)
    np.save(f'matrix_rank_train_{data}1', result_arr)