import pandas as pd
import numpy as np



data_dollar_bars = pd.read_parquet('/Users/liuhaoran/Documents/program/afml/bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet')
data_dollar_bars['timestamp'] = pd.to_datetime(data_dollar_bars['timestamp'])
data_dollar_bars = data_dollar_bars.set_index('timestamp')

data_dollar_bars.index = pd.to_datetime(data_dollar_bars.index).tz_localize(None)

# data_dollar_bars = data_dollar_bars[['open', 'high', 'low', 'close', 'volume']]

print(data_dollar_bars.shape)
display(data_dollar_bars.head(1))

def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        # 计算正向和负向的累计偏离
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

log_returns = np.log(data_dollar_bars['close']).diff()
h_threshold = daily_vol.mean()
tEvents = getTEvents(log_returns, h=daily_vol[-1])
print(len(tEvents))



# 3.1 每日波动率估算
def getDailyVol(close, span0=100):
    # 1. 寻找 24 小时前的索引位置
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    
    # 2. 过滤掉无法回溯 24 小时的初始样本
    df0 = df0[df0 > 0]
    
    # 3. 创建“当前时间”对“过去时间”的映射表
    df0 = pd.Series(close.index[df0 - 1], 
                   index=close.index[close.shape[0] - df0.shape[0]:])
    
    # 4. 计算 24 小时收益率
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
    
    # 5. 使用指数加权移动平均计算标准差（即波动率）
    df0 = df0.ewm(span=span0).std()
    
    return df0


# 3.2 三重屏障标签法
def applyPtSlOnT1(close, events, ptSl, molecule):
    # 1. 根据当前处理的样本子集（molecule）提取事件
    events_ = events.loc[molecule]
    
    # 2. 初始化输出，默认终点为垂直屏障 t1
    out = events_[['t1']].copy(deep=True)
    
    # 3. 计算止盈位 (Profit Taking)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs
        
    # 4. 计算止损位 (Stop Loss)
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index) # NaNs
        
    # 5. 核心逻辑：遍历每一个事件，寻找最早触碰屏障的时间点
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1] # 提取入场到结束的时间段价格
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # 计算回报率
        
        # 记录第一个触碰止损的时间
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min() 
        # 记录第一个触碰止盈的时间
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
        
    return out

# 3.3 确定第一个屏障被触发的时间
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    # 1) 获取目标波动率 (target)
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet] # 过滤掉波动率过小的信号点
    
    # 2) 获取 t1 (最大持仓期/垂直屏障)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
        
    # 3) 构建事件对象，在 t1 上应用止盈止损
    side_ = pd.Series(1., index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, \
                       axis=1).dropna(subset=['trgt'])
    
    # 调用并行处理函数 mpPandasObj 来执行 applyPtSlOnT1
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), \
                      numThreads=numThreads, close=close, events=events, ptSl=[ptSl, ptSl])
    
    # 取止盈、止损、垂直屏障三者中最早发生的那个时刻
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min 忽略 nan
    events = events.drop('side', axis=1)
    
    return events

# 3.4 添加一个垂直屏障 (Adding a Vertical Barrier)
# 1. 寻找每个下单点 tEvents 加上 numDays 后的时间戳在 close.index 中的位置
t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))

# 2. 确保索引位置没有超出 close 数据的总长度（防止越界）
t1 = t1[t1 < close.shape[0]]

# 3. 创建一个 Series：
#    - 值 (Values): 是查找到的过期时刻的时间戳
#    - 索引 (Index): 是对应的下单点时间戳
t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at end



# 3.5 侧与投注大小标签
def getBins(events, close):
    # 1) 价格与事件对齐
    # 提取所有有效的触发时间点，确保没有空值
    events_ = events.dropna(subset=['t1'])
    
    # 获取入场时间点和触发时间点的并集，并去除重复
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    
    # 将价格序列重新索引到这些时间点，缺失值通过后向填充（bfill）处理
    px = close.reindex(px, method='bfill')
    
    # 2) 创建输出对象
    out = pd.DataFrame(index=events_.index)
    
    # 计算从入场时刻到触发时刻（止盈/止损/垂直屏障）的回报率
    # 这里使用了 .values 以确保对齐，防止索引匹配导致的计算问题
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index].values - 1
    
    # 根据回报率的符号打标签：正收益为 1，负收益为 -1
    out['bin'] = np.sign(out['ret'])
    
    return out

# 拓展 getEvents 以支持元标签
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) 获取目标波动率并过滤
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet] # minRet 过滤
    
    # 2) 获取 t1 (最大持仓期)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
        
    # 3) 构建事件对象，应用垂直屏障上的止损
    if side is None:
        # 如果不提供方向，默认设为做多(1.0)，且止盈止损对称
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        # 如果提供了方向（元标签模式），则使用提供的方向和原始 ptSl
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
        
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    # 调用多线程处理函数
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), \
                      numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    
    # 确定最早触发的时间戳
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min 忽略 nan
    
    # 如果是原始模式，删除 side 列
    if side is None:
        events = events.drop('side', axis=1)
        
    return events


# 3.7 拓展 getBins 以支持元标签 (Meta-Labeling)
def getBins(events, close):
    '''
    计算事件的结果（如果提供了 side 信息，则包含方向信息）。
    events 是一个 DataFrame，包含：
    - index: 事件的开始时间
    - t1: 事件的结束时间（垂直屏障）
    - trgt: 事件的目标波动率
    - side (可选): 原始策略预测的持仓方向
    '''
    # 1) 价格与事件对齐
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    # 2) 创建输出对象
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index].values - 1
    
    # --- 元标签处理的核心逻辑 ---
    if 'side' in events_:
        out['ret'] *= events_['side'] # 如果提供了方向，计算策略的实际盈亏
        
    out['bin'] = np.sign(out['ret']) # 初步根据盈亏符号打标
    
    # 在元标签模式下，只要不赚钱（亏损或平手），标签一律设为 0
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        
    return out

# 3.8 抛弃不必要标签
def dropLabels(events, minPct=.05):
    # 应用权重，删除样本不足的标签
    while True:
        # 1. 计算当前各类别标签（-1, 0, 1）的占比
        df0 = events['bin'].value_counts(normalize=True)
        
        # 2. 停止条件：
        #    - 如果占比最小的类别也超过了 minPct (默认5%)
        #    - 或者剩下的类别总数已经少于 3 个
        if df0.min() > minPct or df0.shape[0] < 3: 
            break
        
        # 3. 打印并删除占比最低的那个类别
        print('dropped label', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
        
    return events


