import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 5.1 加权函数
def getWeights(d, size):
    # thres > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show() # 原图中为 mpl.show()，通常需先 import matplotlib.pyplot as plt
    return

if __name__ == '__main__':
    # 运行原图中的两个测试用例
    plotWeights(dRange=[0, 1], nPlots=11, size=6)
    plotWeights(dRange=[1, 2], nPlots=11, size=6)



import numpy as np
import pandas as pd
# 5.2 标准分数阶微分法
def fracDiff(series, d, thres=0.01):
    '''
    使用拓展窗口计算分数阶微分，并处理 NaN 值
    Note 1: 如果 thres=1，则不跳过任何初选计算
    Note 2: d 可以是任何正的分数，不一定限制在 [0, 1]
    '''
    # 1) 为最长的序列计算权重 (需用到 5.1 中的 getWeights 函数)
    w = getWeights(d, series.shape[0])
    
    # 2) 根据权重损失阈值确定初始跳过的计算量
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    
    # 3) 将权重应用到数值上
    df = {}
    for name in series.columns:
        # 填充缺失值并删掉开头
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            # 排除非有限数值
            if not np.isfinite(series.loc[loc, name]):
                continue 
            
            # 执行点积运算 (核心微分计算)
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.iloc[:iloc + 1])[0, 0]
            
        df[name] = df_.copy(deep=True)
    
    df = pd.concat(df, axis=1)
    return df


# 5.3 新型定宽窗口分数阶微分法
def fracDiff_FFD(series, d, thres=1e-5):
    '''
    定宽窗口分数阶微分 (Fast Fractional Differentiation)
    Note 1: thres 决定了权重的截断值，进而决定了固定窗口的宽度
    Note 2: d 可以是任何正的分数
    '''
    # 1) 为最长序列计算固定权重的窗口
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    
    # 2) 将权重应用到数值上
    df = {}
    for name in series.columns:
        # 填充缺失值
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        
        for iloc1 in range(width, seriesF.shape[0]):
            # 确定窗口的起点和终点
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            
            # 排除非有限数值 (NaN 或 Inf)
            if not np.isfinite(series.loc[loc1, name]):
                continue 
            
            # 执行定宽窗口的点积运算
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
            
        df[name] = df_.copy(deep=True)
    
    df = pd.concat(df, axis=1)
    return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 5.4 寻找通过ADF测试的最大D值
def plotMinFFD():
    path, instName = './', 'ES1_Index_Method12'
    # 存储统计结果的 DataFrame
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    
    # 读取数据（这里你可以替换成你的 BTC 数据路径）
    df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)
    
    # 在 0 到 1 之间尝试 11 个不同的 d 值
    for d in np.linspace(0, 1, 11):
        # 1. 对数处理并重采样为日线（降低噪声）
        df1 = np.log(df0[['Close']]).resample('1D').last() 
        
        # 2. 调用之前的 fracDiff_FFD 进行分数阶微分
        df2 = fracDiff_FFD(df1, d, thres=.01)
        
        # 3. 计算微分序列与原始序列的相关性（衡量“记忆”保留程度）
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        
        # 4. 执行 ADF 平稳性检验
        df2 = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
        
        # 5. 存储结果
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]
    
    # 将结果存为 CSV 并绘图
    out.to_csv(path + instName + '_testMinFFD.csv')
    
    # 绘图：左轴显示 ADF 统计量，右轴显示相关系数
    ax = out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.savefig(path + instName + '_testMinFFD.png')
    plt.show()
    return

