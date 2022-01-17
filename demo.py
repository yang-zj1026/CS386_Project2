import numpy as np
import torch
from models.sal_losses import auc_judd
from utils import auc_judd_np, auc_judd_2

# a = np.random.randn(1, 112, 112)
# b = np.random.randn(1, 112, 112)
# print(auc_shuff(a, b, b))

# c = torch.randn((1, 1, 112, 112))
# d = torch.randn((1, 1, 112, 112))
# a1 = auc_judd(c, d)

c1 = np.random.randn(112, 112)
d1 = np.random.randn(112, 112)
a1 = auc_judd_2(c1, d1)
a2 = auc_judd_np(c1, d1)

print(a1, a2)