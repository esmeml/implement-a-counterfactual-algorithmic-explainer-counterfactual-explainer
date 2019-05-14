import numpy as np
from scipy import optimize
from typing import Callable


np.seterr(all='raise')

def mad(data: np.ndarray):
    """Compute median absolute deviation"""
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    return mad

def distance(x: np.ndarray, xp: np.ndarray, mad: np.ndarray, weights: np.ndarray):
    """compute distance between 2 vectors"""
    abs_diff = np.abs(x - xp) * weights
    d = np.sum(abs_diff / mad)
    return d


def make_loss_function(x: np.ndarray, 
                yp: float, 
                balance: float, 
                predict: Callable[[np.ndarray], float], 
                mad: np.ndarray,
                weights: np.ndarray = None) -> Callable[[np.ndarray], float]:
    """Creates a loss function taking one array as input and returning a float"""
    if weights is None:
        weights = np.ones(x.shape)

    def loss(xp: np.ndarray) -> float:
        """Compute loss of the counterfactual"""
        return balance * ((predict(xp) - yp) ** 2) + distance(x, xp, mad, weights)
    return loss

def random_sample(size, min=1, max=0):
    return (max - min) * np.random.random_sample(size) - min

def counterfactual(model: Callable[[np.ndarray], float],
        input_example: np.ndarray,
        desired_output: float, 
        epsilon: float=0.01, 
        init_vector: np.ndarray=None,
        weights: np.ndarray=None, 
        mad: np.ndarray=None,
        n_counterfactuals: int=1,
        initial_balance: float=0.0001,
        minimize_function: Callable[[Callable[[np.ndarray], float], np.ndarray], np.ndarray]=None):
    """Returns a counterfactual i.e. an input close to the initial input \
        that gives the desired output when used as input for the model"""
    if minimize_function != None:
        minimize = minimize_function
    else:
        # default minimize function is from scipy.optimize with default parameters 
        def minimize(loss, xp):
            return optimize.minimize(loss, xp).x

    predict = model
    # initial counterfactual chosen at random
    shape = input_example.shape
    max, min = np.ones(shape), np.zeros(shape)
    xp = random_sample(shape, min, max)
    if init_vector is not None:
        xp = init_vector
    if mad is None:
        mad = np.ones(shape)
    if weights is None:
        weights = np.ones(shape)
    mad[mad == 0] = 1e-19
    # balance between desired output precision and closeness to the initial example
    balance = initial_balance
    # TODO: search for the best balance to be improved
    # 'best' is the closest to the initial to the initial input that gives the desired output
    alpha = 10
    i = 0
    loss = make_loss_function(input_example, desired_output, balance, predict, mad, weights)
    # print(i, distance(input_example, xp, mad, weights), balance, predict(xp), distance(input_example, xp, mad, weights))
    xp = minimize(loss, xp)
    while np.abs(predict(xp) - desired_output) > epsilon:
        balance *= alpha
        loss = make_loss_function(input_example, desired_output, balance, predict, mad, weights)
        # print(i, distance(input_example, xp, mad, weights), balance, predict(xp), distance(input_example, xp, mad, weights))
        xp = minimize(loss, xp)

        min = np.minimum(min, xp)
        max = np.maximum(max, xp)
        # xp = random_sample(shape, min, max)
        i += 1

    if n_counterfactuals == 1:
        return xp
    else:
        res = ()
        loss = make_loss_function(input_example, desired_output, balance, predict, mad, weights)
        for _ in range(n_counterfactuals):
            xp = np.random.random_sample(input_example.shape)
            xp = minimize(loss, xp)
            res += (xp,)
        return res

if __name__ == "__main__":
    from sklearn import ensemble
    from sklearn import datasets
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    import lime
    import lime.lime_tabular
    import matplotlib.pyplot as plt

    np.random.seed(13)

    boston = datasets.load_boston()

    # data = boston.data
    # data = data / data.max(axis=0)

    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    median = np.median(x_train, axis=0)
    mad = np.median(np.abs(x_train - median), axis=0)
    mad[mad == 0] = 1e-19

    x_train /= mad
    x_test /= mad

    model = ensemble.GradientBoostingRegressor()
    model.fit(x_train, y_train)

    score = r2_score(y_test, model.predict(x_test))
    print("R^2: %.4f" % score)

    categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, mode='regression')

    ind = 0
    print(x_test[ind], y_test[ind] * 1.2)
    # print(y_test)
    input_example = x_test[ind]
    desired_output = y_test[ind] * 1.2
    init_vector = x_test[ind + 1]

    def predict(x):
        return model.predict([x])[0]

    exp = explainer.explain_instance(input_example, model.predict, num_features=len(input_example))
    # print(exp.as_map())
    weights = np.ones(input_example.shape)
    for k, coeffs in exp.as_map().items():
        for i, v in coeffs:
            weights[i] = np.abs(v)

    res = counterfactual(predict, x_test[ind], desired_output, epsilon=1, init_vector=init_vector, weights=weights)

    print(res, predict(res))
    plt.subplot(211)
    plt.bar(range(len(res)), res, width=0.4, align='edge')
    plt.bar(range(len(input_example)), input_example, width=-0.4, align='edge')
    plt.xticks(range(len(res)), boston.feature_names)
    plt.subplot(212)
    plt.bar(range(len(res)), res - input_example)
    plt.xticks(range(len(res)), boston.feature_names)
    plt.xlabel(f"Changes output from {y_test[ind]} to {predict(res):.2f}.")
    plt.show()
