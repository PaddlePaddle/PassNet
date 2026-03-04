import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 / 8.0
        tmp_1 = in_1.to(device(type='cuda', index=0))
        tmp_2 = tmp_0 + tmp_1
        tmp_0 = tmp_1 = None
        tmp_3 = torch._C._log_api_usage_once('python.nn_module')
        tmp_3 = None
        return (tmp_2,)