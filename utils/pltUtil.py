import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline


def make_line_interp(old_y, old_x=None, new_num=1000):
    '''
    传入的是list，传出也为list
    '''
    if old_x == None:
        old_x = range(0, len(old_y))
    new_x = np.linspace(min(old_x), max(old_x), new_num)
    func = interp1d(old_x, old_y, kind='cubic')
    new_y = func(new_x)
    return new_x, new_y
    pass


def make_line_smooth(old_y, old_x=None, new_num=1000):
    '''
    传入的是list，传出也为list
    '''
    if old_x == None:
        old_x = range(0, len(old_y))
    new_x = np.linspace(min(old_x), max(old_x), new_num)
    # 插值，参数s为smoothing factor
    func = UnivariateSpline(old_x, old_y, s=1)
    new_y = func(new_x)
    return new_x, new_y
    pass
