from scipy.special import comb # Python 3 中 comb 移到了 scipy.special


# 6.1 袋装分类器的准确性
# 参数设置：
# N: 分类器（树）的总数
# p: 单个分类器预测正确的概率（这里 1/3 意味着比随机猜还差一点）
# k: 分类器的类别数
N, p, k = 100, 1./3, 3.

p_ = 0
# 循环计算二项分布的累积概率
# xrange 替换为 Python 3 的 range
for i in range(0, int(N/k) + 1):
    p_ += comb(N, i) * (p**i) * ((1-p)**(N-i))

# 输出结果：
# p: 单体准确率
# 1-p_: 整个集成系统的准确率
print(f"Single Classifier Accuracy: {p}")
print(f"Ensemble Accuracy: {1 - p_}")


# 6.2 设置随机森林的3种方法
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 方法 1：标准随机森林 (Standard RF)
# 这种方法最简单，但无法直接利用我们自定义的唯一性 (avgU) 进行样本量限制
clf0 = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', 
                               criterion='entropy')

# 方法 2：使用决策树作为基估计器的装袋分类器 (Bagging Classifier with DTC)
# 这里的关键是 max_samples=avgU，它利用了第 4 章算出的平均唯一性来控制每一棵树抽样的规模
clf1 = DecisionTreeClassifier(criterion='entropy', max_features='auto', 
                               class_weight='balanced')
clf1 = BaggingClassifier(base_estimator=clf1, n_estimators=1000, max_samples=avgU)

# 方法 3：使用单棵树的随机森林作为基估计器的装袋分类器
# 这是德普拉多最推荐的“双重包装”法，能更好地控制特征随机性和样本唯一性
clf2 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, 
                               class_weight='balanced_subsample')
clf2 = BaggingClassifier(base_estimator=clf2, n_estimators=1000, max_samples=avgU, 
                          max_features=1.)



# #6.1 采样机制
# 为什么装袋算法（Bagging）是基于有放回地随机采样？如果没有放回地抽样，装袋算法仍会减少预测值的方差吗？

# 6.2 标签重叠与唯一性
# 假设你的训练集是基于高度重叠的标签（如第 4 章所述，具有较低的唯一性）。

# 这是否使装袋算法易于过拟合，还是只是无效？为什么？

# 金融应用中的**袋外准确性（OOB Accuracy）**一般可靠吗？为什么？

# 6.3 集成估算器构建
# 建立一个集成估算器，其中基估算器是一个决策树。

# 这种集成如何与随机森林相区别？

# 使用 sklearn 生成一个像随机森林一样运行的装袋分类器，你需要设置什么参数，又该如何设置？

# 6.4 随机森林的参数关系
# 考虑一个随机森林，其包含的决策树的数量、所用的特征数量以及观测数量三者之间的关系：

# 你可以构想一下在随机森林里所需最少的决策树数量和所用的特征数量之间的关系吗？

# 决策树的数量对于所用的特征数量来说是否太少？

# 决策树的数量对于可观测数量来说是否太多？

# 6.5 验证方法对比
# 袋外准确性（OOB Accuracy）与分层 k 折（含数据打乱）交叉验证的准确性有何不同？


# 7.1 清洗训练集中的观测值
def getTrainTimes(t1, testTimes):
    '''
    根据测试集的时间段，找出并剔除训练集中与其重叠的观测值
    - t1.index: 观测值开始的时间
    - t1.value: 观测值结束的时间（离场时刻）
    - testTimes: 测试集观测值的时间
    '''
    trn = t1.copy(deep=True)
    # 注意：在较新版本的 pandas 中，iteritems() 建议改为 items()
    for i, j in testTimes.items():
        # 1. 训练集观测值在测试集期间开始
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        
        # 2. 训练集观测值在测试集期间结束
        df1 = trn[(i <= trn) & (trn <= j)].index
        
        # 3. 训练集观测值包含了整个测试集段
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        
        # 剔除这三种重叠情况的交集
        trn = trn.drop(df0.union(df1).union(df2))
        
    return trn



# 7.2 训练样本中观测结果的禁制
def getEmbargoTimes(times, pctEmbargo):
    # Get embargo time for each bar
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg


# --------------------------------------------------

testTimes = pd.Series(mbrg[dt1], index=[dt0])  # include embargo before purge
trainTimes = getTrainTimes(t1, testTimes)
testTimes = t1.loc[dt0:dt1].index

# 7.3 当观测值重叠时的交叉验证
class PurgedKFold(_BaseKFold):
    """
    扩展 KFold 类，以处理跨越时间间隔的标签。
    训练集会清洗掉与测试标签时间间隔重叠的观测值。
    假设测试集是连续的（shuffle=False），中间没有训练样本。
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in \
                       np.array_split(np.arange(X.shape[0]), self.n_splits)]
        
        for i, j in test_starts:
            t0 = self.t1.index[i]  # 测试集开始时间
            test_indices = indices[i:j]
            # 找到测试集结束后的索引位置（用于清洗逻辑）
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            
            # 1. 寻找左侧训练集（在测试集开始之前结束的样本）
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            
            # 2. 寻找右侧训练集（在测试集结束并经过禁运期之后开始的样本）
            if maxT1Idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            
            yield train_indices, test_indices


# 7.4 使用数据清洗的k折类别
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# 注意：假设你已经定义或导入了之前提取的 PurgedKFold 类
# from your_module import PurgedKFold 

def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', 
            t1=None, cv=None, cvGen=None, pctEmbargo=None):
    
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    
    # 如果没有提供现成的生成器，则现场创建一个 PurgedKFold
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo) # purged
        
    score = []
    # 遍历交叉验证的每一折
    for train, test in cvGen.split(X=X):
        # 1. 拟合模型：使用清洗后的训练索引，并传入对应的样本权重
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], 
                      sample_weight=sample_weight.iloc[train].values)
        
        # 2. 评估模型
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, 
                               sample_weight=sample_weight.iloc[test].values, 
                               labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, 
                                    sample_weight=sample_weight.iloc[test].values)
            
        score.append(score_)
        
    return np.array(score)


# 8.1 MDI特征重要性
import pandas as pd
import numpy as np

def featImpMDI(fit, featNames):
    # 基于样本内（In-Sample）平均不纯度减少的特征重要性
    # fit: 已经训练好的集成模型（如随机森林）
    # featNames: 特征名称列表
    
    # 1. 提取每一棵决策树对各特征的重要性
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    
    # 2. 转换为 DataFrame，行是树的索引，列是特征
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    
    # 3. 处理 max_features=1 的情况
    # 因为如果一棵树没用到某个特征，sklearn 会记为 0，这会拉低平均值
    # 我们将其替换为 NaN，这样在计算 mean() 时只考虑用到了该特征的树
    df0 = df0.replace(0, np.nan) 
    
    # 4. 计算平均重要性（mean）和标准差（std）
    # 注意：这里的 std 进行了缩放处理
    imp = pd.concat({'mean': df0.mean(), 
                     'std': df0.std() * df0.shape[0]**-.5}, axis=1)
    
    # 5. 归一化，使平均重要性之和为 1
    imp /= imp['mean'].sum()
    
    return imp

# 8.2 mda特征重要性
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')

    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]

        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob,
                                   sample_weight=w1.values,
                                   labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred,
                                        sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column

            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob,
                                          sample_weight=w1.values,
                                          labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred,
                                               sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)

    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1. - scr1)

    imp = pd.concat({'mean': imp.mean(),
                     'std': imp.std() * imp.shape[0]**-0.5}, axis=1)

    return imp, scr0.mean()


# 代码片段 8.3：SFI 的实现方法
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], 
                      scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0]**-.5
    return imp



# 8.4 正交特征的运算
def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim+1], eVec.iloc[:, :dim+1]
    return eVal, eVec

# ---------------------------------------------------------

def orthoFeats(dfX, varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    # standardize
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    
    return dfP


# 8.5 运算特征重要性于逆向pca排序之间的加权肯德尔值
import numpy as np
from scipy.stats import weightedtau
featImp = np.array([.55, .33, .07, .05]) # feature importance
pcRank = np.array([1, 2, 4, 3]) # PCA rank
weightedtau(featImp, pcRank**-1.)[0]

# 8.6创造一组合成数据集
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features,
                                      n_informative=n_informative, n_redundant=n_redundant,
                                      random_state=0, shuffle=False)
    
    # pandas>=3.0: DatetimeIndex 不再支持 periods/end 参数，使用 date_range 生成
    # 同时将 end 先滚动到最后一个工作日，避免周末导致长度 < periods
    end = pd.Timestamp.today().normalize()
    if end.weekday() >= 5:  # 5=Sat, 6=Sun
        end = end - pd.tseries.offsets.BDay(1)
    df0 = pd.date_range(end=end, periods=n_samples, freq=pd.tseries.offsets.BDay())
    
    trnsX, cont = pd.DataFrame(trnsX, index=df0), \
                  pd.Series(cont, index=df0).to_frame('bin')
    
    df0 = ['I_' + str(i) for i in range(n_informative)] + \
          ['R_' + str(i) for i in range(n_redundant)]
    
    df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
    trnsX.columns = df0
    
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    
    return trnsX, cont


# 8.7在任意方法中调用特重要性
def featImportance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1., numThreads=24,
                   pctEmbargo=0, scoring='accuracy', method='SFI', minWLeaf=0., **kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from mpEngine import mpPandasObj
    
    n_jobs = (-1 if numThreads > 1 else 1) # run 1 thread with ht_helper in diracl
    
    # #1) prepare classifier, cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced', min_weight_fraction_leaf=minWLeaf)
    
    clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators,
                            max_features=1., max_samples=max_samples, oob_score=True, n_jobs=n_jobs)
    
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_
    
    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
                      t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring).mean()
    
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
                              t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring)
        
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring,
                      cvGen=cvGen).mean()
        
        clf.n_jobs = 1 # parallelize auxFeatImpSFI rather than clf
        imp = mpPandasObj(auxFeatImpSFI, ('featNames', trnsX.columns), numThreads,
                          clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
        
    return imp, oob, oos


# 8.8 调用所有组件
def testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000,
             n_samples=10000, cv=10):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features - n_informative - n_redundant
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'],
             'max_samples': [1.]}
    
    jobs, out = (dict(izip(dict0, i)) for i in product(*dict0.values())), []
    kargs = {'pathOut': './testFunc/', 'n_estimators': n_estimators,
             'tag': 'testFunc', 'cv': cv}
    
    for job in jobs:
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_' + '%.2f'%job['minWLeaf'] + '_' + str(job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
        plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
        
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos}); df0.update(job)
        out.append(df0)
        
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    out.to_csv(kargs['pathOut'] + 'stats.csv')
    return

# 8.9 特征重要性绘图
def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    mpl.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'})
    
    if method == 'MDI':
        mpl.xlim([0, imp.sum(axis=1).max()])
        mpl.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
        
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2,
                i.get_y() + i.get_height() / 2, j, ha='center', va='center',
                color='black')
        
    mpl.title('tag=' + tag + ' | simNum=' + str(simNum) + ' | oob=' + str(round(oob, 4)) +
              ' | oos=' + str(round(oos, 4)))
    mpl.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    mpl.clf(); mpl.close()
    return

