import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn import functional as F

class Model(nn.Module):
  """Model is the architecture of the network that 
  we need to optimize as part of QRT Data Challenge
  Challenge is restricting model architecture (we cannot do deeper neural network).

  Attributes:
    A is the orthogonal matrix that allows to generate the explicative factors
    B is the matrix that projects explicative factors and predicts the next time step return
  """
  def __init__(self, p=0.5, D_size=250, F_size=10):
    super().__init__()
    self.A = nn.utils.parametrizations.orthogonal(nn.Linear(F_size, D_size, bias=False))  ## need to reverse input and output in order to have the correct orthogonal matrix
    self.B = nn.Linear(F_size,1, bias=False)
    self.drop_layer = nn.Dropout(p=p)

  def forward(self,x):
    F_m = F.linear(x, self.A.weight.T, self.A.bias)
    drop_F_m = self.drop_layer(F_m)
    return self.B(drop_F_m)