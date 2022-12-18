import torch
from collections import defaultdict

class MyDataset(torch.utils.data.Dataset):
  """This is a key component of the proposed architecture.
  Model is optimized to maximize challenge metric.
  The effort here is to have one sample corresponding to all the returns for all the stocks at a given time step (date)
  """
  def __init__(self,x, y):
    self.x=torch.tensor(x.to_numpy(),dtype=torch.float32)
    self.y=torch.tensor(y.to_numpy(),dtype=torch.float32)
    self.date = [i[0] for i in x.index]
    self.set_date = list(set(self.date))
    x_by_date = defaultdict(list)
    y_by_date = defaultdict(list)

    for k, i in enumerate(self.date):
      x_by_date[i].append(self.x[k].view((1, -1)))
      y_by_date[i].append(self.y[k].view((1, -1)))

    self.dict_data = {
      i: (torch.concat(x_by_date[i], axis=0), torch.concat(y_by_date[i], axis=0))
      for i in self.set_date
    } ## we create a dict # key = date , value = (past returns for all stocks before date, ground truth return to predict)

  def __len__(self):
    return len(self.set_date)
  
  def __getitem__(self,idx):
    date = self.set_date[idx]
    return self.dict_data[date]