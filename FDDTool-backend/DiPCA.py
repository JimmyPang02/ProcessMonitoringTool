# _*_ coding:utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
from sklearn.model_selection import KFold

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 归一化数据


def normalize(*args):
    """归一化数据
    Returns:
        np.array: 归一化后的数据
    """
    X_normal = args[0]
    X_normal_mean = np.mean(X_normal, axis=0)
    X_normal_std = np.std(X_normal, axis=0)
    X_normal_row, X_normal_col = X_normal.shape
    X_normal_center = (X_normal - X_normal_mean) / X_normal_std

    if len(args) == 2:
        X_new = args[1]
        X_new_row, X_new_col = X_new.shape
        X_new_center = (X_new - X_normal_mean) / X_normal_std
        return (X_normal_center, X_new_center)

    return X_normal_center


def pc_number(X):
    """计算主成分个数

    Args:
        X (np.array): 输入矩阵 

    Returns:
        int: 主成分个数
    """
    U, S, V = np.linalg.svd(X)
    if S.shape[0] == 1:
        i = 1
    else:
        i = 0
        var = 0
        while var < 0.85*sum(S*S):
            var = var+S[i]*S[i]
            i = i + 1
    return i


def fit_DiPCA(X, s, a):
    """DiPCA建模

    Args:
        X (np.array): 输入矩阵
        s (int): 滞后阶数
        a (int): 主成分个数

    Returns:
        P (np.array): 静态负荷矩阵
        W (np.array): 动态负荷矩阵
        Theta_hat (np.array): 系数矩阵
        PHI_v (np.array): 动态综合指标控制限
        PHI_s (np.array): 静态指标控制限
        phi_v_lim (float): 动态综合指标控制限
        phi_s_lim (float): 静态指标控制限
    """
    n = X.shape[0]
    m = X.shape[1]
    N = n - s
    Xe = X[s:N+s, :]
    alpha = 0.01
    level = 1-alpha
    P = np.zeros((m, a))
    W = np.zeros((m, a))
    T = np.zeros((n, a))
    w = np.ones(m)
    w = w / np.linalg.norm(w, ord=2)

    if s > 0:
        # Dynamic Inner Modeling
        l = 0
        while l < a:
            iterative_error = 1000
            iterative_nums = 1000
            t = np.dot(X, w)
            temp = np.dot(X, w)
            while (iterative_error > 1e-5) & (iterative_nums > 0):
                beta = np.ones((s))
                for i in range(s):
                    beta[i] = np.dot(t[i:N+i-1].T, t[s:N+s-1])
                beta = beta / np.linalg.norm(beta, ord=2)
                w = np.zeros(m)
                for i in range(s):
                    w = w + beta[i]*(np.dot(X[s:N+s-1, :].T, t[i:N+i-1]) +
                                     np.dot(X[i:N+i-1].T, t[s:N+s-1]))
                w = w / np.linalg.norm(w, ord=2)
                t = np.dot(X, w)
                iterative_error = np.linalg.norm((t-temp), ord=2)
                temp = t
                iterative_nums -= 1
            # p = np.dot(X.T, t)/np.dot(t.T, t)
            p = X.T @ t/(t.T@t)
            t = np.array([t]).T
            p = np.array([p]).T
            X = X - np.dot(t, p.T)
            P[:, l] = p[:, 0]
            W[:, l] = w
            T[:, l] = t[:, 0]
            l = l+1

        TT = T[0:N, :]
        for j in range(1, s):
            TT = np.c_[TT, T[j:(N+j), :]]
        Theta_hat = np.dot(np.dot(np.linalg.inv(
            np.dot(TT.T, TT)), TT.T), T[s:N+s, :])
        V = T[s:N+s, :] - TT @ Theta_hat
        Xe = Xe-np.dot(np.dot(TT, Theta_hat), P.T)
       # Calculate the control limit
        a_v = pc_number(V)
        _, Sv, Pv = np.linalg.svd(V)
        Pv = Pv.T
        Pv = Pv[:, 0:a_v]
        lambda_v = (1/(N-1)*np.diag(Sv[0:a_v]**2))
        Tv2_lim = a_v * (N ** 2 - 1) / (N * (N - a_v)) * \
            scipy.stats.f.ppf(level, a_v, N-a_v)
        if a_v == a:
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T)
            phi_v_lim = Tv2_lim
        else:
            gv = 1/(N-1)*sum(Sv[a_v:a]**4)/sum(Sv[a_v:a]**2)
            hv = (sum(Sv[a_v:a]**2)**2)/sum(Sv[a_v:a]**4)
            Qv_lim = gv*scipy.stats.chi2.ppf(level, hv)
            PHI_v = np.dot(np.dot(Pv, np.linalg.inv(lambda_v)), Pv.T) / \
                Tv2_lim + (np.identity(len(Pv@Pv.T))-Pv@Pv.T)/Qv_lim
            SS_v = 1/(N-1)*V.T@V
            g_phi_v = np.trace((SS_v@PHI_v)@(SS_v@PHI_v)) / \
                (np.trace(SS_v@PHI_v))
            h_phi_v = (np.trace(SS_v@PHI_v)**2) / \
                np.trace((SS_v@PHI_v)@(SS_v@PHI_v))
            phi_v_lim = g_phi_v*scipy.stats.chi2.ppf(level, h_phi_v)
    a_s = pc_number(Xe)
    _, Ss, Ps = np.linalg.svd(Xe)
    Ps = Ps.T
    Ps = Ps[:, 0:a_s]
    lambda_s = 1/(N - 1) * np.diag(Ss[0:a_s] ** 2)
    m = Ss.shape[0]
    # gs = 1 / (N - 1) * sum(Ss[a_s:m] ** 4) / sum(Ss[a_s:m] ** 2)
    # hs = (sum(Ss[a_s:m] ** 2) ** 2) / sum(Ss[a_s:m] ** 4)
    Ts2_lim = scipy.stats.chi2.ppf(level, a_s)
    # Qs_lim = gs*scipy.stats.chi2.ppf(level,hs)
    if a_s == m:
        PHI_s = np.dot(np.dot(Ps, np.linalg.inv(lambda_s)), Ps.T)
        phi_s_lim = Ts2_lim
    else:
        gs = 1/(N-1)*sum(Ss[a_s:m]**4)/sum(Ss[a_s:m]**2)
        hs = (sum(Ss[a_s:m]**2)**2)/sum(Ss[a_s:m]**4)
        Qs_lim = gs*scipy.stats.chi2.ppf(level, hs)
        PHI_s = np.dot(np.dot(Ps, np.linalg.inv(lambda_s)), Ps.T) / \
            Ts2_lim + (np.identity(len(Ps@Ps.T))-Ps@Ps.T)/Qs_lim
        SS_s = 1/(N-1)*Xe.T@Xe
        g_phi_s = np.trace((SS_s@PHI_s)@(SS_s@PHI_s))/(np.trace(SS_s@PHI_s))
        h_phi_s = (np.trace(SS_s@PHI_s)**2)/np.trace((SS_s@PHI_s)@(SS_s@PHI_s))
        phi_s_lim = g_phi_s*scipy.stats.chi2.ppf(level, h_phi_s)
    return P, W, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim


def test_DiPCA(X, P, W, Theta, s, PHI_s, PHI_v):
    """DiPCA测试,计算动态综合指标和静态指标,都是由T2指标和Q指标组成的

    Args:
        X (np.array): 输入矩阵
        P (np.array): 静态负荷矩阵
        W (np.array): 动态负荷矩阵
        Theta (np.array): 系数矩阵
        s (int): 滞后阶数
        PHI_s (np.array): 静态指标控制限
        PHI_v (np.array): 动态综合指标控制限

    Returns:
        phi_v_index (np.array): 动态综合指标
        phi_s_index (np.array): 静态指标
    """
    n = X.shape[0]
    N = n - s
    R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    if s > 0:
        T = np.dot(X, R)
        TTs = T[s:N+s, :]
        TT = T[0:N, :]
        i = 1
        while i < s:
            Ts = T[i:N+i, :]
            TT = np.c_[TT, Ts]
            i = i + 1
        TTshat = np.dot(TT, Theta)
    phi_v_index = np.zeros(N)
    phi_s_index = np.zeros(N)
    k = s
    while k < s+N:
        if s > 0:
            temp = TTs[k-s, :] - TTshat[k-s, :]
            temp = np.array([temp])
            v = temp.T
            phi_v_index[k-s] = np.dot(np.dot(v.T, PHI_v), v)
            e = X[k-s, :].T - np.dot(P, TTshat[k-s, :].T)
        else:
            e = X[k-s, :].T
        # Ts_index[k-s] = np.dot(np.dot(e.T, Mst), e)
        # Qs_index[k-s] = np.dot(np.dot(e.T, Msq), e)
        phi_s_index[k-s] = np.dot(np.dot(e.T, PHI_s), e)
        k = k+1
    # return phi_v_index,Ts_index,Qs_index
    return phi_v_index, phi_s_index


def predict_DiPCA(X, P, W, Theta_hat, s):
    """DiPCA预测,预测下一时刻的数据

    Args:
        X (np.array): 输入矩阵
        P (np.array): 静态负荷矩阵
        W (np.array): 动态负荷矩阵
        Theta_hat (np.array): 系数矩阵
        s (int): 滞后阶数

    Returns:
        x_predict_d (np.array): 预测矩阵
    """
    n = X.shape[0]
    N = n - s
    # a = P.shape[1]
    x_predict_d = np.zeros(X.shape, dtype=float)
    R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
    if s > 0:
        T = np.dot(X, R)
        TT = T[0:N, :]
        i = 1
        while i < s:
            Ts = T[i:N+i, :]
            TT = np.c_[TT, Ts]
            i = i + 1
        TTshat = np.dot(TT, Theta_hat)
        x_predict_d[s:, :] = TTshat@P.T
    return x_predict_d


def cv_DiPCA(X, s_range, a_range, fold):
    """DiPCA交叉验证,选取最优的滞后阶数和主成分个数

    Args:
        X (np.array): 输入矩阵
        s_range (int): 滞后阶数范围
        a_range (int): 主成分个数范围
        fold (int): 交叉验证折数

    Returns:
        s (int): 最优滞后阶数
        a (int): 最优主成分个数
    """
    kf = KFold(n_splits=fold, shuffle=False)
    press = np.zeros((s_range, a_range, fold), float)
    for i in range(s_range):
        for j in range(a_range):
            count = 0
            for train_index, valid_index in kf.split(X):
                count += 1
                X_train, X_valid = X[train_index], X[valid_index]
                P, W, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim = fit_DiPCA(
                    X_train, i+1, j+1)  # 建模
                X_predict = predict_DiPCA(X_valid, P, W, Theta_hat, i+1)  # 预测
                press[i][j][count -
                            1] = np.linalg.norm(X_valid-X_predict, ord=2)**2/X_valid.shape[0]
    press = np.sum(press, axis=2)
    (s, a) = np.where(press == np.min(press))  # 选择press最小作为s,a
    s += 1
    a += 1
    return int(s), int(a)


def visualization_DiPCA(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim):
    """
        DiPCA可视化
        目前主要是2个监控指标，包括动态综合指标φv,静态综合指标φs
        参数
        ----------
    """

    plt.figure(figsize=(9.6, 6.4), dpi=300)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(phi_v_index)
    ax1.plot(phi_v_lim*np.ones(len(phi_v_index)), 'r--')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('$\phi_v$')
    ax1.set_yscale('log')  # 设置对数尺度
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(phi_s_index)
    ax2.plot(phi_s_lim*np.ones(len(phi_s_index)), 'r--')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('$\phi_s$')
    ax2.set_yscale('log')  # 设置对数尺度
    plt.show()


def runDiPCA(x_train, x_test, s, a):
    """DiPCA主函数

    Args:
        x_train (np.array): 训练集
        x_test (np.array): 测试集
        s (int): 滞后阶数
        a (int): 主成分个数

    Returns:
        phi_v_index (np.array): 动态综合指标
        phi_s_index (np.array): 静态综合指标
    """

    print(x_train.shape, x_test.shape)
    X_train, X_test = normalize(x_train, x_test)

    P, W, Theta_hat, PHI_v, PHI_s, phi_v_lim, phi_s_lim = fit_DiPCA(
        X_train, s, a)  # 建模
    phi_v_index, phi_s_index = test_DiPCA(
        X_test, P, W, Theta_hat, s, PHI_s, PHI_v)  # 测试

    return phi_v_index, phi_s_index, phi_v_lim, phi_s_lim
    # visualization_DiPCA(phi_v_index, phi_s_index, phi_v_lim, phi_s_lim)  # 监测结果可视化


if __name__ == '__main__':
    x_train = np.loadtxt("../../dataset/csv/T2.csv", delimiter=',')
    x_test = np.loadtxt("../../dataset/csv/Set5_2.csv", delimiter=',')
    s = 2  # s为滞后阶数
    a = 5  # a为主成分数量
    runDiPCA(x_train, x_test, s, a)
