# 输入：
#  data //观测数据集
#  n //拟合模型所需要的最少的数据点个数
#  k //最大允许迭代次数
#  t //阈值，若数据点带入模型所得误差小于 t，则认为该数据点属于该模型的一致集
#  d //阈值，若当前模型的一致集中数据点的个数多于 d，则认为该一致集已经足够好

# 输出：
# bestFit //拟合出来的模型参数，若为空则表明拟合失败
# iterations = 0
# bestFit = null
# bestErr = something really large
# while iterations < k do
#  maybeInliers := n randomly selected values from data
#  maybeModel := model parameters fitted to maybeInliers
#  alsoInliers := empty set
#  for every point in data not in maybeInliers do //计算 maybeModel 的一致集
#  if point fits maybeModel with an error smaller than t
#  add point to alsoInliers
#  end if
#  end for
#  if the number of elements in alsoInliers is > d then
#  // 这意味着我们可能已经找到了一个很好的模型
#  // 把该模型从当前一致集中拟合出来
#  betterModel := model parameters fitted to all points in maybeInliers and alsoInliers
#  thisErr := a measure of how well betterModel fits these points
#  if thisErr < bestErr then //完成输出模型及其误差更新
#  bestFit := betterModel
#  bestErr := thisErr
#  end if
#  end if
#  increment iterations
# end while


from copy import copy
import numpy as np
from numpy.random import default_rng
rng = default_rng()


class RANSAC:
    def __init__(self, n=2, k=100, t=20, d=5, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X, y):
        for _ in range(self.k):
            # X = [[9],[2],[3],[4],[7]]-> ids = [2, 1, 3, 0, 4]
            ids = rng.permutation(X.shape[0])   # 生成随机序列，长度为X.shape[0]，数据类型为整数
            # n = 3 -> ids[:3] -> [2, 1, 3]
            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(
                X[maybe_inliers], y[maybe_inliers])
            # 使用模型预测的值与真实值比较，小于阈值的为内点
            # bool数组，True为内点
            thresholded = (
                self.loss(y[ids][self.n:], maybe_model.predict(
                    X[ids][self.n:])) < self.t
            )
            #  留下在thresholded（预测正确）中为True的索引，即内点的索引
            inlier_ids = ids[self.n:][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(
                    X[inlier_points], y[inlier_points])

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = better_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 使用最小二乘法，求出参数
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


if __name__ == "__main__":

    regressor = RANSAC(model=LinearRegressor(),
                       loss=square_error_loss, metric=mean_square_error)

    data = [(0, 0.9), (2, 2.0), (3, 6.5), (4, 2.9), (5, 8.8), (6, 3.95), (8, 5.03),
            (10, 5.97), (12, 7.1), (13, 1.2), (14, 8.2), (16, 8.5), (18, 10.1)]
    X = np.array([point[0] for point in data]).reshape(-1, 1)
    y = np.array([point[1] for point in data]).reshape(-1, 1)

    regressor.fit(X, y)
    import matplotlib.pyplot as plt
    # 绘制散点图
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)
    plt.scatter(X, y)
    # 绘制直线
    line = np.linspace(-1, 20, num=100).reshape(-1, 1)
    plt.plot(line, regressor.predict(line), c="peru")
    plt.show()
