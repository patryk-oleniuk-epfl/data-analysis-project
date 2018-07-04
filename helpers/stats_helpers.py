import numpy as np


def exponential_density(x, lambda_):
    return lambda_ * np.exp(- lambda_ * x)


def normalize(data):
    return np.array(data) / sum(data)


def exponential_mle(data):
    return len(data) / sum(data)


def sample_exponential(data, mle):
    values_to_evaluate = sorted(list(set(data)))
    samples = [exponential_density(v, mle) for v in values_to_evaluate]
    return samples


def pareto_mle(data):
    beta_hat = min(data)
    alpha_hat = len(data) / (np.sum(np.log(np.array(data))) + len(data) * np.log(beta_hat))
    return (alpha_hat, beta_hat)


def poisson_mle(data):
    return sum(data) / len(data)


def poisson_density(x, lambda_):
    return np.exp(-lambda_) * (pow(lambda_, x) / np.math.factorial(x))


def pareto_density(x, alpha, beta):
    return alpha * (pow(beta, alpha) / pow(x, alpha))


def log_logistic_density(x, alpha, beta):
    return ((beta / alpha) * pow(x / alpha, beta - 1)) / pow((1 + pow(x / alpha, beta)), 2)
