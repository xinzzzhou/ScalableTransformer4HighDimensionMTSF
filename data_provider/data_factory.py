from data_provider.data_loader import Dataset_Custom, Dataset_Crime, Dataset_Wiki
from torch.utils.data import DataLoader

data_dict = {
    'crime': Dataset_Crime,
    'traffic': Dataset_Custom,
    'wiki': Dataset_Wiki
}

def data_provider(args, flag, logger=None):
    Data = data_dict[args.data]
    if args.task_name != 'statistic':
        timeenc = 0 if args.embed != 'timeF' else 1
    else:
        timeenc = 0
        
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True
    batch_size = args.batch_size 
    freq = args.freq
    print(args.data)
    data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    data_topk_path=args.data_topk_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    timeenc=timeenc,
    freq=freq,
    k=args.k,
    seasonal_patterns=args.seasonal_patterns)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
