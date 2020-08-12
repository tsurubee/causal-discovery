import pandas as pd
from scipy import stats

def chi_square(data_matrix, X, Y, Z, **kwargs):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    Parameters
    ----------
    data: numpy.ndarray
        The data matrix

    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.

    Returns
    -------
    p_value: float
        The p_value, i.e. the probability of observing the computed chi-square
        statistic
    """

    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )
    data = pd.DataFrame(data_matrix)

    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0)
        )
    else:
        chi = 0
        dof = 0
        for _, df in data.groupby(Z):
            c, _, d, _ = stats.chi2_contingency(
                df.groupby([X, Y]).size().unstack(Y, fill_value=0)
            )
            chi += c
            dof += d
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    return p_value
