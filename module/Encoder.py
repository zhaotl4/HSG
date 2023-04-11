from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from module.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"


