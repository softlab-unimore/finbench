import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# feature_cols = ['high','low','close','open','to','vol']

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)

if __name__ == "__main__":
    # path1 = "../data/csi300.pkl"
    # df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
    #prev_date_num Indicates the number of days in which stock correlation is calculated
    # prev_date_num = 20
    # date_unique=df1['dt'].unique()
    # stock_trade_data=date_unique.tolist()
    # stock_trade_data.sort()
    # stock_num=df1.code.unique().shape[0]
    #dt is the last trading day of each month
    # dt=['2022-11-30','2022-12-30']
    # for i in ['2020','2021','2022']:
    #     for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    #         stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
    #         dt.append(stock_m[-1])
    # df1['dt']=df1['dt'].astype('datetime64')

    path1 = '../data/sp500.csv'
    df1 = pd.read_csv(path1, encoding='utf-8')
    train_start = '2010-01-01'
    train_end = '2017-12-31'

    feature_cols = ['high', 'low', 'close', 'open', 'volume']
    df1 = df1.loc[(df1['date'] >= train_start) & (df1['date'] <= train_end)]
    cols = ['instrument', 'date'] + feature_cols
    df1 = df1[cols]

    prev_date_num = 20
    date_unique = df1['date'].unique()
    stock_trade_data = date_unique.tolist()
    stock_trade_data.sort()
    # stock_num = df1.instrument.unique().shape[0]
    df1['date'] = df1['date'].astype('datetime64')

    dt = df1.groupby(df1['date'].dt.to_period('M'))['date'].max().tolist()
    # feature_cols = ['high', 'low', 'close', 'open', 'to', 'vol']


    for i in range(len(dt)):
        df2 = df1.copy()
        end_data = dt[i].strftime("%Y-%m-%d")
        start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
        df2 = df2.loc[df2['date'] <= end_data]
        df2 = df2.loc[df2['date'] >= start_data]
        code = sorted(list(set(df2['instrument'].values.tolist())))
        test_tmp = {}
        for j in tqdm(range(len(code))):
            df3 = df2.loc[df2['instrument'] == code[j]]
            y = df3[feature_cols].values
            if y.T.shape[1] == prev_date_num:
                test_tmp[code[j]] = y.T
        t1 = time.time()
        result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
        result=result.fillna(0)
        for i in range(0, result.shape[0] - 1):
            result.iloc[i,i]=1
        t2 = time.time()
        print('time cost', t2 - t1, 's')
        result.to_csv("../data/relation/"+str(end_data)+".csv")
