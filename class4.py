import pandas as pd
import numpy as np



# 4.1 估算一个标签的唯一性
def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    计算每一根 bar 上的并发事件数量。
    - molecule[0] 是计算权重的第一个事件的日期
    - molecule[-1] 是计算权重的最后一个事件的日期
    任何在 t1[molecule].max() 之前开始的事件都会影响计数。
    '''
    # 1) 筛选出跨越 [molecule[0], molecule[-1]] 期间的相关事件
    # 填充缺失的 t1（未平仓事件），默认终点为序列最后一个索引
    t1 = t1.fillna(closeIdx[-1]) 
    
    # 筛选在当前“分子”起始时间之后结束的事件
    t1 = t1[t1 >= molecule[0]] 
    
    # 筛选在当前“分子”最晚结束时间之前开始的事件
    t1 = t1.loc[:t1[molecule].max()] 
    
    # 2) 计算跨越每一根 bar 的事件数
    # 找到受影响的时间范围在 closeIdx 中的位置
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    
    # 初始化一个计数器序列，索引为受影响的 bars
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    
    # 遍历每一个事件的生命周期 (入场 tIn 到 出场 tOut)
    for tIn, tOut in t1.items():
        # 在该事件持仓期间，给对应的 bars 计数加 1
        count.loc[tIn:tOut] += 1
        
    # 只返回当前处理子集（molecule）范围内的数据
    return count.loc[molecule[0]:t1[molecule].max()]



# 对一个标签的平均唯一性进行估算
def mpSampleW(t1, numCoEvents, molecule):
    # 根据事件的生命周期衍生出平均唯一性
    # 1) 初始化一个 Series 来存储权重 (唯一性)
    wght = pd.Series(index=molecule)
    
    # 2) 遍历当前子集 (molecule) 中的每一个事件
    for tIn, tOut in t1.loc[wght.index].items():
        # 计算该事件在持仓期间（tIn 到 tOut）每一根 bar 的唯一性倒数
        # 唯一性 = 1 / 同时在场事件数
        # 然后取均值作为该样本的平均唯一性
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
        
    return wght

# --- 以下是图片下半部分的调用逻辑 ---

# 1. 计算每一根 bar 上的并发事件数
numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', events.index), numThreads, \
                         closeIdx=close.index, t1=events['t1'])

# 2. 清洗 numCoEvents：去除重复项并对齐原始价格序列索引
numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
numCoEvents = numCoEvents.reindex(close.index).fillna(0)

# 3. 计算每个样本的平均唯一性 (Time-Weighted Uniqueness)
out['tW'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, \
                       t1=events['t1'], numCoEvents=numCoEvents)


# 4.3构建一个指示矩阵
def getIndMatrix(barIx, t1):
    # 构建一个指示矩阵
    # 1. 初始化一个全为 0 的 DataFrame
    # 每一行是一根 bar 的索引，每一列代表一个交易事件（用 0 到 N-1 编号）
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    
    # 2. 遍历每一个事件及其对应的起始时间 (t0) 和 结束时间 (t1)
    for i, (t0, t1_) in enumerate(t1.items()):
        # 将该事件生命周期 [t0, t1_] 覆盖的所有 bars 设为 1
        indM.loc[t0:t1_, i] = 1.
        
    return indM

# 4.4 计算平均唯一性    
def getAvgUniqueness(indM):
    # 从指示矩阵计算平均唯一性
    
    # 1. 计算每一根 bar 上的并发事件数 (Concurrency)
    # 对矩阵按行求和，算出每个时刻有多少个事件在场
    c = indM.sum(axis=1) 
    
    # 2. 计算每个事件在每一时刻的即时唯一性 (Uniqueness)
    # 用指示矩阵除以并发数。如果某时刻有 2 个事件，则每个事件在该时刻的唯一性为 0.5
    u = indM.div(c, axis=0)
    
    # 3. 计算所有事件的平均唯一性
    # 排除掉值为 0 的部分（即事件不在场的时间段），对所有在场时刻取均值
    avgU = u[u > 0].mean()
    
    return avgU


# 4.5 序列性引导程序的返还样本
def seqBootstrap(indM, sLength=None):
    # 通过序列自助法生成一个样本子集
    
    # 1) 如果未指定抽样长度，默认抽取和原样本量相等的次数
    if sLength is None: 
        sLength = indM.shape[1]
    
    phi = [] # 存储已抽中的样本索引
    
    while len(phi) < sLength:
        avgU = pd.Series()
        
        # 2) 遍历所有候选样本，计算“如果把该样本加入已选集合”后的平均唯一性
        for i in indM:
            # 尝试将样本 i 加入当前已选的 phi 列表中
            indM_ = indM[phi + [i]]
            
            # 计算加入后，这组样本中最后一个样本（即 i）的平均唯一性
            # getAvgUniqueness 返回的是所有样本的唯一性，这里我们只取最后一个
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
            
        # 3) 根据唯一性计算抽样概率
        # 唯一性越高的样本，被抽中的概率越大
        prob = avgU / avgU.sum()
        
        # 4) 根据概率分布随机抽取一个样本，并加入 phi
        phi += [np.random.choice(indM.columns, p=prob)]
        
    return phi

# 4.6 序列性引导程序示例
def main():
    # 1. 模拟三个事件，定义它们的入场时间 (index) 和离场时间 (values)
    # 事件 0: [0, 2], 事件 1: [2, 3], 事件 2: [4, 5]
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4]) 
    
    # 2. 生成对应的 bar 索引（简单的 0 到 5）
    barIx = range(t1.max() + 1)
    
    # 3. 构建指示矩阵 (Indicator Matrix)
    indM = getIndMatrix(barIx, t1)
    
    # --- 实验 1：标准随机抽样 (Standard Bootstrap) ---
    # 随机抽取 3 次（允许重复）
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(f"标准随机抽样选中的索引: {phi}")
    # 计算这组随机样本的平均唯一性
    std_uniqueness = getAvgUniqueness(indM[phi]).mean()
    print(f"标准抽样的平均唯一性: {std_uniqueness}")
    
    # --- 实验 2：序列自助法 (Sequential Bootstrap) ---
    # 使用德普拉多提出的序列算法进行抽样
    phi = seqBootstrap(indM)
    print(f"序列自助法选中的索引: {phi}")
    # 计算这组序列样本的平均唯一性
    seq_uniqueness = getAvgUniqueness(indM[phi]).mean()
    print(f"序列抽样的平均唯一性: {seq_uniqueness}")
    
    return

# 4.7生成随机t1序列
def getRndT1(numObs, numBars, maxH):
    # 生成一个随机的 t1 序列（用于测试）
    # numObs: 想要生成的样本（事件）数量
    # numBars: 模拟的总时间长度（Bar 的数量）
    # maxH: 每个事件最大可能的持有时间
    
    t1 = pd.Series()
    for i in range(numObs):
        # 1. 随机生成入场时间点 (ix)
        ix = np.random.randint(0, numBars)
        
        # 2. 随机生成持有时间，并算出离场时间点 (val)
        # 持有时间在 [1, maxH] 之间随机
        val = ix + np.random.randint(1, maxH)
        
        # 3. 存入序列：入场时间作为 Index，离场时间作为 Value
        t1.loc[ix] = val
        
    # 按入场时间排序后返回
    return t1.sort_index()


# 4.8标准与序列引导程序得出的唯一性
def auxMC(numObs, numBars, maxH):
    # 这是一个并行化辅助函数，用于对比标准抽样与序列抽样的唯一性
    
    # 1. 生成随机的交易事件序列 t1
    t1 = getRndT1(numObs, numBars, maxH)
    
    # 2. 生成对应的 bar 索引
    barIx = range(t1.max() + 1)
    
    # 3. 构建指示矩阵 (Indicator Matrix)
    indM = getIndMatrix(barIx, t1)
    
    # 4. 执行标准随机抽样 (Standard Bootstrap) 并计算平均唯一性
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    
    # 5. 执行序列自助法抽样 (Sequential Bootstrap) 并计算平均唯一性
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    
    # 6. 返回两种方法得出的唯一性结果
    return {'stdU': stdU, 'seqU': seqU}




# 注意：mpEngine 是书中的多进程工具包，通常对应你环境中的 mpPandasObj 或类似的并行封装
from mpEngine import processJobs, processJobs_ 

# 4.9 多进程蒙特卡洛实验
def mainMC(numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24):
    # 启动多进程蒙特卡洛实验
    # numObs: 每次实验生成的样本数
    # numBars: 模拟的时间轴长度
    # maxH: 最大持仓时间
    # numIters: 实验循环次数（模拟次数）
    # numThreads: 并行使用的 CPU 核心数
    
    jobs = []
    # 1. 构造任务清单
    for i in range(int(numIters)):
        # 每一个任务都是一次 auxMC 实验
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
    
    # 2. 根据线程数决定运行方式
    if numThreads == 1:
        # 单线程运行（调试用）
        out = processJobs_(jobs)
    else:
        # 多进程并行运行
        out = processJobs(jobs, numThreads=numThreads)
    
    # 3. 输出统计结果
    # 这里会打印出 stdU (标准唯一性) 和 seqU (序列唯一性) 的 均值、标准差、分位数等
    print(pd.DataFrame(out).describe())
    
    return

# 4.10 利用绝对回报归因法对样本权重进行判定
def mpSampleW(t1, numCoEvents, close, molecule):
    # 利用收益率归因法衍生样本权重
    # 1. 计算对数收益率，确保收益是可加的
    ret = np.log(close).diff() 
    
    # 2. 初始化权重序列
    wght = pd.Series(index=molecule)
    
    # 3. 遍历当前子集中的每一个事件
    for tIn, tOut in t1.loc[wght.index].items():
        # 核心公式：将该事件在场期间的每一根 bar 的收益率除以当时的并发事件数
        # 这意味着这根 bar 的收益被平均分配到了所有在场的交易中
        # 最后的 sum() 就是该交易分得的“总收益贡献”
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
        
    # 4. 取绝对值，因为我们关注的是波动的幅度（无论是涨是跌）
    return wght.abs()

# --- 调用逻辑 ---

# 1. 调用并行插件计算原始权重 'w'
out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, \
                      t1=events['t1'], numCoEvents=numCoEvents, close=close)

# 2. 归一化处理
# 确保所有样本的权重总和等于样本总数，这样平均权重保持为 1.0
out['w'] *= out.shape[0] / out['w'].sum()


# 时间衰减系数的应用
def getTimeDecay(tW, clfLastW=1.0):
    # 对观察到的唯一性应用分段线性衰减
    # 最新的观测权重=1.0，最旧的观测权重=clfLastW
    
    # 1. 计算累计唯一性
    clfW = tW.sort_index().cumsum()
    
    # 2. 计算衰减斜率
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
        
    # 3. 计算最终权重
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0 # 确保权重不为负
    
    # print(const, slope) # 可选：打印参数
    return clfW