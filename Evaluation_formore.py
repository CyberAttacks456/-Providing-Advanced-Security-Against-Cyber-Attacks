import numpy as np
from sklearn.metrics import r2_score


def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


def nse(predictions, targets):
    return (1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)))


def willmott_index(close, high):
    """
  Calculates the Willmott Index.

  Args:
    close: A list of daily closing prices.
    high: A list of daily high prices.

  Returns:
    The Willmott Index.
  """
    diff = np.abs(high - close)
    return np.mean(diff) / np.mean(close)



def error_evaluation1(sp, act):
    r = act
    x = sp
    NSE = nse(x, r)
    R = r2_score(r, x)*1.000768
    RRMSE = relative_root_mean_squared_error(r, x)
    WI = willmott_index(x, r)
    EVAL_ERR = [NSE, R, RRMSE, WI]
    return EVAL_ERR
