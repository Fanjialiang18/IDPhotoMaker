import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

import torch
import torch.nn as nn
import torch.nn.functional as F


class smaple:
    def __init__(self):
        print(torch.cuda.is_available())


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)