import numpy as np
def levelset_to_c_binary(U_sample, c_high, c_low):
    """
    Binary level-set mapping:
        if U > 0 -> c_high
        else     -> c_low
    """
    U_sample = np.asarray(U_sample)
    return np.where(U_sample > 0.0, c_high, c_low)