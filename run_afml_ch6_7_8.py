import os
from itertools import product
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from scipy.special import comb
from scipy.stats import weightedtau

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as mpl


# ============================================================
# Fractional Differentiation (FFD) - 第5章的实现，内嵌到第6-8章脚本
# ============================================================


def getWeights_FFD(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    定宽窗口 FFD：生成权重，直到 |w_k| < thres 为止。
    返回形状为 (width, 1)
    """
    w: List[float] = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series: pd.DataFrame, d: float, thres: float = 1e-5) -> pd.DataFrame:
    """
    Fast Fractional Differentiation (定宽窗口)：
    y_t = sum_{k=0..width} w_k * x_{t-k}
    """
    w = getWeights_FFD(d, thres=thres)
    width = len(w) - 1

    out: Dict[str, pd.Series] = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype="float64")

        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = float(np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0])

        out[name] = df_.copy(deep=True)

    return pd.concat(out, axis=1)


# ============================================================
# PurgedKFold + CV scoring - 第7章
# ============================================================


class PurgedKFold:
    """
    Purged K-fold：为避免训练集与测试集 label 区间重叠造成的信息泄露。
    同时支持 embargo：测试集结束后的一段样本不进入训练。

    约定：
    - X.index 是样本的 t0（事件起点）
    - t1 是 pd.Series，index 与 X.index 相同，values 为事件终点（thru date）
    - split 按 X 的时间顺序进行（shuffle=False）
    """

    def __init__(self, n_splits: int, t1: pd.Series, pctEmbargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series")
        if not isinstance(t1.index, pd.DatetimeIndex):
            # 允许任意 index，但下面的比较要求可排序
            t1 = t1.sort_index()
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.pctEmbargo = float(pctEmbargo)

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if len(X) != len(self.t1):
            raise ValueError("X and t1 must have same length")
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X.index and t1.index must match exactly")

        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        # embargo 以“样本个数”估算，不用真实时间长度，和书里一致的近似方式
        embargo = int(n_samples * self.pctEmbargo)

        # 连续切分
        test_splits = np.array_split(indices, self.n_splits)
        for test_idx in test_splits:
            if len(test_idx) == 0:
                continue
            i0, i1 = int(test_idx[0]), int(test_idx[-1]) + 1  # half-open
            test_indices = indices[i0:i1]

            test_start = X.index[i0]
            test_end = X.index[i1 - 1]

            # embargo 区间（测试集结束后排除）
            embargo_end = min(n_samples, i1 + embargo)
            embargo_indices = indices[i1:embargo_end] if embargo_end > i1 else np.array([], dtype=int)

            # 计算 purging：训练事件与测试区间重叠则剔除
            # 重叠条件：t0_train <= test_end 且 t1_train >= test_start
            train_candidates = np.setdiff1d(indices, np.concatenate([test_indices, embargo_indices]), assume_unique=True)

            t0_train = X.index[train_candidates]
            t1_train = self.t1.iloc[train_candidates].values
            overlap = (t0_train <= test_end) & (t1_train >= test_start)

            train_indices = train_candidates[~overlap]
            yield train_indices, test_indices


def cvScore(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    scoring: str = "neg_log_loss",
    cvGen: PurgedKFold = None,
) -> np.ndarray:
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("wrong scoring method.")
    if cvGen is None:
        raise ValueError("cvGen must be provided")

    scores: List[float] = []
    for train_idx, test_idx in cvGen.split(X=X):
        X0, y0, w0 = X.iloc[train_idx, :], y.iloc[train_idx], sample_weight.iloc[train_idx]
        X1, y1, w1 = X.iloc[test_idx, :], y.iloc[test_idx], sample_weight.iloc[test_idx]

        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            s = -log_loss(y1, prob, sample_weight=w1.values, labels=fit.classes_)
        else:
            pred = fit.predict(X1)
            s = accuracy_score(y1, pred, sample_weight=w1.values)
        scores.append(float(s))
    return np.array(scores, dtype="float64")


# ============================================================
# Feature importance - 第8章（简化但与书一致的思想）
# ============================================================


def featImpMDI(fit, featNames: List[str]) -> pd.DataFrame:
    """
    MDI：In-sample 平均不纯度减少的特征重要性（Bagging/集成的每棵树取 feature_importances_ 再平均）
    """
    # BaggingClassifier.estimators_ 每个元素是基估计器（这里是 RandomForest 或 DecisionTree）
    df0 = {i: getattr(tree, "feature_importances_", None) for i, tree in enumerate(fit.estimators_)}
    df0 = {i: v for i, v in df0.items() if v is not None}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames

    # 将 0 视为“没用到该特征”（书里用 0->NaN 处理 max_features=1 时的稀疏问题）
    df0 = df0.replace(0, np.nan)

    imp = pd.concat(
        {
            "mean": df0.mean(),
            "std": df0.std() * df0.shape[0] ** (-0.5),
        },
        axis=1,
    )
    imp = imp / imp["mean"].sum()
    return imp


def featImpMDA(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    cv: int,
    pctEmbargo: float,
    scoring: str = "neg_log_loss",
) -> Tuple[pd.DataFrame, float]:
    """
    MDA：out-of-sample permutation importance（在 purged CV 的每一折里做单列打乱）
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("wrong scoring method.")

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    scr0 = pd.Series(dtype="float64")
    scr1 = pd.DataFrame(columns=X.columns, dtype="float64")

    for i, (train_idx, test_idx) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train_idx, :], y.iloc[train_idx], sample_weight.iloc[train_idx]
        X1, y1, w1 = X.iloc[test_idx, :], y.iloc[test_idx], sample_weight.iloc[test_idx]

        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            base = -log_loss(y1, prob, sample_weight=w1.values, labels=fit.classes_)
        else:
            pred = fit.predict(X1)
            base = accuracy_score(y1, pred, sample_weight=w1.values)
        scr0.loc[i] = float(base)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            # 单列 permutation：numpy shuffle 需要可写数组
            vals = X1_[j].to_numpy(copy=True)
            np.random.shuffle(vals)
            X1_[j] = vals
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                s = -log_loss(y1, prob, sample_weight=w1.values, labels=fit.classes_)
            else:
                pred = fit.predict(X1_)
                s = accuracy_score(y1, pred, sample_weight=w1.values)
            scr1.loc[i, j] = float(s)

    # 按书里公式构造重要性（以 OOS 分数下降幅度为依据）
    # 对于 neg_log_loss：scr0/scr1 都是“负的 logloss”，越接近 0 越好
    # permute 后变差 => scr1 更负 => -scr1 变大，从而重要性为正
    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / (-scr1)
    else:
        imp = imp / (1.0 - scr1)

    imp = pd.concat(
        {
            "mean": imp.mean(),
            "std": imp.std() * imp.shape[0] ** (-0.5),
        },
        axis=1,
    )
    return imp, float(scr0.mean())


def featImpSFI(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    cv: int,
    pctEmbargo: float,
    scoring: str = "neg_log_loss",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SFI：只用单个特征训练，得到每个特征对应的 out-of-sample score。
    """
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    imp = pd.DataFrame(index=X.columns, columns=["mean", "std"], dtype="float64")
    scr = {}

    for featName in X.columns:
        X1 = X[[featName]]
        scores = cvScore(clf, X=X1, y=y, sample_weight=sample_weight, scoring=scoring, cvGen=cvGen)
        imp.loc[featName, "mean"] = float(scores.mean())
        imp.loc[featName, "std"] = float(scores.std() * scores.shape[0] ** (-0.5))
        scr[featName] = scores

    # 归一化（和书里“相对重要性”的处理一致）
    imp["mean"] = imp["mean"] / imp["mean"].abs().sum()
    return imp, pd.Series({k: float(v.mean()) for k, v in scr.items()})


# ============================================================
# 8.1-8.4: orthogonal features + feature importance helpers
# ============================================================


def get_eVec(dot: pd.DataFrame, varThres: float) -> Tuple[pd.Series, pd.DataFrame]:
    """
    从 dot(相关度/协方差)矩阵得到特征向量，并根据累计解释方差降维。
    对应 class6_7_8.py: get_eVec。
    """
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]

    eVal_series = pd.Series(eVal, index=[f"PC_{i+1}" for i in range(eVal.shape[0])])
    eVec_df = pd.DataFrame(eVec, index=dot.index, columns=eVal_series.index)
    eVec_df = eVec_df.loc[:, eVal_series.index]

    cumVar = eVal_series.cumsum() / eVal_series.sum()
    dim = int(cumVar.values.searchsorted(varThres))
    eVal_series = eVal_series.iloc[: dim + 1]
    eVec_df = eVec_df.iloc[:, : dim + 1]
    return eVal_series, eVec_df


def orthoFeats(dfX: pd.DataFrame, varThres: float = 0.95) -> pd.DataFrame:
    """
    正交特征（Orthogonal Features）：把高度相关的特征投影到主成分子空间。
    对应 class6_7_8.py: orthoFeats。
    """
    # 这里按 class6_7_8.py 的写法保留 axis=1（尽管这与常见写法不完全一致）
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    _, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    dfP = pd.DataFrame(dfP, index=dfX.index, columns=eVec.columns)
    return dfP


def weighted_kendall(featImp: np.ndarray, pcRank: np.ndarray) -> float:
    """
    对应 class6_7_8.py: 8.5 加权 Kendall 值示例。
    """
    return float(weightedtau(featImp, pcRank ** -1.0)[0])


# ============================================================
# 8.6-8.8: synthetic data + unified featImportance interface
# ============================================================


def getTestData(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    8.6：构造人工二分类数据，用于验证不同特征重要性方法是否能恢复真实重要特征。
    对应 class6_7_8.py: getTestData。
    """
    trnsX, cont = make_classification(n_samples=n_samples,n_features=n_features,n_informative=n_informative,n_redundant=n_redundant,random_state=0,shuffle=False,)
    # pandas>=3.0: DatetimeIndex 不再支持 periods/end 参数，使用 date_range 生成日期序列
    df0 = pd.date_range(
        end=pd.Timestamp.today(),
        periods=n_samples,
        freq=pd.tseries.offsets.BDay(),
    )
    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame("bin")

    feat_names = [f"I_{i}" for i in range(n_informative)] + [f"R_{i}" for i in range(n_redundant)]
    feat_names += [f"N_{i}" for i in range(n_features - len(feat_names))]
    trnsX.columns = feat_names

    cont["w"] = 1.0 / cont.shape[0]
    cont["t1"] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


def auxFeatImpSFI(
    featNames: List[str],
    clf,
    trnsX: pd.DataFrame,
    cont: pd.DataFrame,
    scoring: str,
    cvGen: PurgedKFold,
) -> pd.DataFrame:
    """
    对应 class6_7_8.py: auxFeatImpSFI（这里用串行实现，避免 mpEngine 依赖）。
    """
    imp = pd.DataFrame(index=featNames, columns=["mean", "std"], dtype="float64")
    for featName in featNames:
        scores = cvScore(
            clf,
            X=trnsX[[featName]],
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen,
        )
        imp.loc[featName, "mean"] = float(scores.mean())
        imp.loc[featName, "std"] = float(scores.std() * scores.shape[0] ** (-0.5))
    return imp


def featImportance(
    trnsX: pd.DataFrame,
    cont: pd.DataFrame,
    n_estimators: int = 1000,
    cv: int = 10,
    max_samples: float = 1.0,
    numThreads: int = 1,
    pctEmbargo: float = 0.0,
    scoring: str = "accuracy",
    method: str = "SFI",
    minWLeaf: float = 0.0,
) -> Tuple[pd.DataFrame, float, float]:
    """
    8.7：统一入口，支持 MDI / MDA / SFI 三种重要性方法。
    返回：
    - imp：重要性 DataFrame（mean/std）
    - oob：袋外分数（BaggingClassifier）
    - oos：purged CV 的平均分数
    对应 class6_7_8.py: featImportance。
    """
    n_jobs = (-1 if numThreads > 1 else 1)

    clf = DecisionTreeClassifier(criterion="entropy",max_features=1,class_weight="balanced",min_weight_fraction_leaf=minWLeaf,)
    clf = BaggingClassifier(estimator=clf,n_estimators=n_estimators,max_features=1.0,max_samples=max_samples,oob_score=True,n_jobs=n_jobs,bootstrap=True,random_state=0,)

    fit = clf.fit(X=trnsX, y=cont["bin"], sample_weight=cont["w"].values)
    oob = float(fit.oob_score_) if hasattr(fit, "oob_score_") else np.nan

    cvGen = PurgedKFold(n_splits=cv, t1=cont["t1"], pctEmbargo=pctEmbargo)
    oos = float(cvScore(it if method == "MDI" else clf,X=trnsX,y=cont["bin"],sample_weight=cont["w"],scoring=scoring if scoring in ["neg_log_loss", "accuracy"] else "accuracy",cvGen=cvGen).mean())

    if method == "MDI":
        imp = featImpMDI(fit, featNames=list(trnsX.columns))
        return imp, oob, oos
    if method == "MDA":
        imp, oos_mda = featImpMDA(clf=clf,X=trnsX,y=cont["bin"],sample_weight=cont["w"],t1=cont["t1"],cv=cv,pctEmbargo=pctEmbargo,scoring="neg_log_loss" if scoring == "neg_log_loss" else "accuracy")
        return imp, oob, float(oos_mda)
    if method == "SFI":
        # SFI：每次只用单特征训练，得到 OOS 分数（通过 cvScore）
        scores_all = cvScore(clf,X=trnsX,y=cont["bin"],sample_weight=cont["w"],scoring="neg_log_loss" if scoring == "neg_log_loss" else "accuracy",cvGen=cvGen).mean()
        oos_sfi = float(scores_all.mean())
        imp = auxFeatImpSFI(featNames=list(trnsX.columns),clf=clf,trnsX=trnsX,cont=cont,scoring="neg_log_loss" if scoring == "neg_log_loss" else "accuracy",cvGen=cvGen)
        return imp, oob, oos_sfi

    raise ValueError(f"Unknown method: {method}")


def plotFeatImportance(
    pathOut: str,
    imp: pd.DataFrame,
    oob: float,
    oos: float,
    method: str,
    tag: str = "tag",
    simNum: int = 0,
) -> None:
    """
    8.9：重要性可视化（对应 class6_7_8.py: plotFeatImportance）
    """
    os.makedirs(pathOut, exist_ok=True)
    mpl.figure(figsize=(10, imp.shape[0] / 5.0))
    imp_sorted = imp.sort_values("mean", ascending=True)

    ax = imp_sorted["mean"].plot(
        kind="barh",
        color="b",
        alpha=0.25,
        xerr=imp_sorted["std"],
        error_kw={"ecolor": "r"},
    )

    if method == "MDI":
        # 画一个“均分”参考线
        mpl.axvline(1.0 / imp_sorted.shape[0], linewidth=1, color="r", linestyle="dotted")

    ax.get_yaxis().set_visible(False)
    for patch, feat in zip(ax.patches, imp_sorted.index):
        ax.text(
            patch.get_width() / 2,
            patch.get_y() + patch.get_height() / 2,
            feat,
            ha="center",
            va="center",
            color="black",
        )

    mpl.title(f"tag={tag} | simNum={simNum} | oob={round(oob,4)} | oos={round(oos,4)}")
    mpl.savefig(os.path.join(pathOut, f"featImportance_{simNum}.png"), dpi=100, bbox_inches="tight")
    mpl.clf()
    mpl.close()


def testFunc(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_estimators: int = 500,
    n_samples: int = 5000,
    cv: int = 5,
    pathOut: str = "./outputs/ch6_7_8_testFunc/",
) -> pd.DataFrame:
    """
    8.8：在人工数据上测试 MDI/MDA/SFI 是否能恢复真实重要特征类型。
    这一步计算量较大，默认给较小规模，避免你本地跑太久。
    """
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {
        "minWLeaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0],
    }

    rows = []
    jobs = list(dict(product(*[dict0[k] for k in dict0.keys()])))[0:0]  # kept to mirror structure
    for combi in product(*dict0.values()):
        sim = dict(zip(dict0.keys(), combi))
        # 按 class5 的示例拼一个简化的 simNum
        simNum = f"{sim['method']}_{sim['scoring']}_mw{sim['minWLeaf']:.2f}_ms{sim['max_samples']}"
        imp, oob, oos = featImportance(trnsX=trnsX,cont=cont,n_estimators=n_estimators,cv=cv,max_samples=sim["max_samples"],numThreads=1,pctEmbargo=0.0,scoring=sim["scoring"],method=sim["method"],minWLeaf=sim["minWLeaf"],)
        plotFeatImportance(pathOut=pathOut,imp=imp,oob=oob,oos=oos,method=sim["method"],tag=sim["method"],simNum=0,)

        # 统计重要性落在 I/R/N 三类上的权重
        df0 = imp[["mean"]].copy()
        df0["type"] = [idx.split("_")[0] for idx in df0.index]
        df0 = df0.groupby("type")["mean"].sum().to_dict()
        df0.update({"oob": oob, "oos": oos})
        df0.update(sim)
        df0["simNum"] = str(simNum)
        rows.append(df0)

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(pathOut, "stats.csv"), index=False)
    return stats


# ============================================================
# Main: 串起 6/7/8 章
# ============================================================


@dataclass(frozen=True)
class Ch6_7_8Config:
    # 输入来自第3-4章输出
    ch34_dir: str = "outputs/ch3_ch4"

    # 价格数据（用于生成第5章的 FFD 特征）
    dollar_bars_parquet: str = "bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet"
    ts_col: str = "timestamp"
    close_col: str = "close"

    # 第5章 d 选择结果（从 csv 读 pVal 最小通过的 d）
    ch5_d_csv: str = "outputs/ch5/BTCUSDT_FFD_testMinFFD.csv"
    fd_thres: float = 0.01

    # 特征构造
    n_lags: int = 5

    # Purged CV 参数（第7章）
    cv_splits: int = 3
    pct_embargo: float = 0.01

    # Bagging 参数（第6章）
    n_estimators: int = 200
    random_state: int = 0

    # 输出
    out_dir: str = "outputs/ch6_7_8"


def load_close_from_raw(cfg: Ch6_7_8Config) -> pd.Series:
    df = pd.read_parquet(cfg.dollar_bars_parquet)
    if cfg.ts_col in df.columns:
        df[cfg.ts_col] = pd.to_datetime(df[cfg.ts_col])
        df = df.set_index(cfg.ts_col)
    close = pd.to_numeric(df[cfg.close_col], errors="raise")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def pick_d_from_ch5_csv(csv_path: str, pval_thres: float = 0.05) -> float:
    df = pd.read_csv(csv_path)
    # d 在 index 列里；csv 第一列是空列名，所以读出来会是 unnamed 或用 df.columns[0]
    d_col = df.columns[0]
    df[d_col] = pd.to_numeric(df[d_col], errors="coerce")
    df = df.dropna(subset=["pVal", d_col])

    df = df.sort_values(d_col)
    passed = df[df["pVal"] < pval_thres]
    if passed.empty:
        # 兜底：取最大 d
        return float(df[d_col].max())
    return float(passed.iloc[0][d_col])


def make_features_from_fd(fd_series: pd.Series, event_index: pd.DatetimeIndex, n_lags: int) -> pd.DataFrame:
    """
    用 FFD 序列构造特征：
    - fd_lag0..fd_lag(n_lags-1)
    - fd_ret_lag1..fd_ret_lag(n_lags-2)
    """
    feats = {}
    for lag in range(n_lags):
        feats[f"fd_lag{lag}"] = fd_series.shift(lag)

    # 简单的变化率特征：fd_ret = fd_t - fd_{t-1}
    fd_ret = fd_series - fd_series.shift(1)
    for lag in range(max(1, n_lags - 1)):
        feats[f"fd_ret_lag{lag}"] = fd_ret.shift(lag)

    X_all = pd.DataFrame(feats)
    X = X_all.reindex(event_index)
    return X


def main() -> None:
    cfg = Ch6_7_8Config()
    print(cfg) #Ch6_7_8Config(ch34_dir='outputs/ch3_ch4', dollar_bars_parquet='bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet', ts_col='timestamp', close_col='close', ch5_d_csv='outputs/ch5/BTCUSDT_FFD_testMinFFD.csv', fd_thres=0.01, n_lags=5, cv_splits=3, pct_embargo=0.01, n_estimators=200, random_state=0, out_dir='outputs/ch6_7_8')
    os.makedirs(cfg.out_dir, exist_ok=True)

    # -----------------------------
    # Load 第3-4章数据
    # -----------------------------
    ch34 = cfg.ch34_dir
    events = pd.read_parquet(os.path.join(ch34, "events.parquet"))
    bins_dropped = pd.read_parquet(os.path.join(ch34, "bins_dropped.parquet"))
    tW = pd.read_parquet(os.path.join(ch34, "tW.parquet"))
    w = pd.read_parquet(os.path.join(ch34, "w.parquet"))

    # events.index == events 的 t0
    # bins_dropped['bin'] 用作分类标签
    y_raw = bins_dropped["bin"]

    # 对齐
    common_index = events.index.intersection(y_raw.dropna().index)
    events = events.loc[common_index]
    y_raw = y_raw.loc[common_index]
    tW = tW.loc[common_index]
    w = w.loc[common_index]

    # 映射标签到 {0,1}，方便 log_loss
    y = (y_raw > 0).astype(int)

    # 样本权重
    sample_weight = w.iloc[:, 0].astype(float)

    # 事件终点 t1，用于 purged CV
    t1 = events["t1"].astype("datetime64[ns]")

    # avgU：第4章的平均唯一性。这里用 tW 的均值近似成标量（用于 max_samples）
    avgU = float(tW.iloc[:, 0].mean())
    avgU = max(1e-6, min(1.0, avgU))

    print("=== Chapter 6/7/8 inputs ===")
    print(f"events: {len(events)}")
    print(f"label distribution (0/1): {y.value_counts().to_dict()}")
    print(f"avgU (for max_samples): {avgU:.6f}")
    print(f"sample_weight range: [{sample_weight.min():.6g}, {sample_weight.max():.6g}]")

    # -----------------------------
    # 第5章：选 d，并生成 FFD 特征
    # -----------------------------
    d = pick_d_from_ch5_csv(cfg.ch5_d_csv, pval_thres=0.05)
    print(f"Picked FFD d = {d:.2f} (based on pVal < 0.05)")

    close = load_close_from_raw(cfg)
    log_close = np.log(close).to_frame("Close").dropna()
    fd = fracDiff_FFD(log_close, d=float(d), thres=cfg.fd_thres)
    fd_series = fd["Close"]

    X = make_features_from_fd(fd_series, event_index=events.index, n_lags=cfg.n_lags)
    # 丢弃特征存在 NaN 的样本（因为 lag）
    valid_mask = X.notna().all(axis=1)
    X = X.loc[valid_mask]
    events = events.loc[valid_mask]
    t1 = events["t1"].astype("datetime64[ns]")
    y = y.loc[valid_mask]
    sample_weight = sample_weight.loc[valid_mask]

    print(f"Feature matrix X: {X.shape[0]} samples, {X.shape[1]} features")

    # -----------------------------
    # 第6章：袋装分类器（double bagging 的思想）
    # -----------------------------
    # 单棵树/标准 RF
    clf0 = RandomForestClassifier(n_estimators=500,criterion="entropy",class_weight="balanced_subsample",random_state=cfg.random_state)

    # bagging + DTC（不使用 avgU 的版本：max_samples=1.0）
    base_dt = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=cfg.random_state)
    clf1 = BaggingClassifier(estimator=base_dt,n_estimators=cfg.n_estimators,max_samples=1.0,max_features=1.0,bootstrap=True,oob_score=True,random_state=cfg.random_state)

    # double bagging：基估计器是一棵退化 RF（n_estimators=1），外层 bagging 控制 max_samples=avgU
    base_rf = RandomForestClassifier(n_estimators=1,criterion="entropy",bootstrap=False,class_weight="balanced_subsample",random_state=cfg.random_state)
    clf2 = BaggingClassifier(estimator=base_rf,n_estimators=cfg.n_estimators,max_samples=avgU,max_features=1.0,bootstrap=True,oob_score=True,random_state=cfg.random_state)

    # -----------------------------
    # 第7章：purged CV 验证
    # -----------------------------
    cvGen = PurgedKFold(n_splits=cfg.cv_splits, t1=t1, pctEmbargo=cfg.pct_embargo)

    # 用 log loss（越大越好，因为我们用 neg_log_loss）
    for name, clf in [("clf0_RF", clf0), ("clf1_BagDTC", clf1), ("clf2_DoubleBagRF_avgU", clf2)]:
        scores = cvScore(clf=clf,X=X,y=y,sample_weight=sample_weight,scoring="neg_log_loss",cvGen=cvGen).mean()
        acc_scores = cvScore(clf=clf,X=X,y=y,sample_weight=sample_weight,scoring="accuracy",cvGen=cvGen).mean()
        print(f"\n{name}:")
        print(f"  OOS neg_log_loss mean={scores.mean():.6f} std={scores.std():.6f}")
        print(f"  OOS accuracy      mean={acc_scores.mean():.6f} std={acc_scores.std():.6f}")

    # 选 clf2 作为后续重要性分析（因为它最贴近第6章的推荐结构）
    clf = clf2

    # 训练一个全样本模型，用于 MDI
    fit_all = clf.fit(X=X, y=y, sample_weight=sample_weight.values)

    # -----------------------------
    # 第8章：特征重要性
    # -----------------------------
    featNames = list(X.columns)
    mdi = featImpMDI(fit_all, featNames=featNames)
    mdi.to_csv(os.path.join(cfg.out_dir, "feat_importance_MDI.csv"))

    mda, oos = featImpMDA(clf=clf,X=X,y=y,sample_weight=sample_weight,t1=t1,cv=cfg.cv_splits,pctEmbargo=cfg.pct_embargo,scoring="neg_log_loss")
    mda.to_csv(os.path.join(cfg.out_dir, "feat_importance_MDA.csv"))

    # SFI 也做（单特征模型），特征数不大才可用
    if len(featNames) <= 25:
        sfi, _ = featImpSFI(clf=clf,X=X,y=y,sample_weight=sample_weight,t1=t1,cv=cfg.cv_splits,pctEmbargo=cfg.pct_embargo,scoring="neg_log_loss")
        sfi.to_csv(os.path.join(cfg.out_dir, "feat_importance_SFI.csv"))
    else:
        sfi = None

    print("\n=== Chapter 8 feature importance saved ===")
    print(f"MDI: {os.path.join(cfg.out_dir, 'feat_importance_MDI.csv')}")
    print(f"MDA: {os.path.join(cfg.out_dir, 'feat_importance_MDA.csv')}")
    if sfi is not None:
        print(f"SFI: {os.path.join(cfg.out_dir, 'feat_importance_SFI.csv')}")

    # -----------------------------
    # 小结输出：打印 Top 特征
    # -----------------------------
    print("\nTop features by MDI mean:")
    print(mdi["mean"].sort_values(ascending=False).head(10))
    print("\nTop features by MDA mean:")
    print(mda["mean"].sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()


    # 可选：按第 8.8 章在人工数据上验证（默认不跑，避免耗时）
    # if os.environ.get("RUN_CH8_TESTFUNC", "0") == "1":
    #     print("\n[Ch8] Running testFunc() on synthetic data...")
    #     _ = testFunc(
    #         n_samples=3000,
    #         n_estimators=200,
    #         cv=3,
    #         n_features=30,
    #         n_informative=8,
    #         n_redundant=8,
    #         pathOut="outputs/ch6_7_8/testFunc/",
    #     )

