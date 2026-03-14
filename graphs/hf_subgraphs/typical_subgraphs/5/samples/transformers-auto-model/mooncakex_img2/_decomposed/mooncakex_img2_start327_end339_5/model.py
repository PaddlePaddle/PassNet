import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_1.transpose(-1, -2)
        tmp_1 = torch.matmul(in_2, tmp_0)
        tmp_0 = None
        tmp_2 = tmp_1 / 8.0
        tmp_1 = None
        tmp_3 = in_0.to(device(type='cuda', index=0))
        tmp_4 = tmp_2 + tmp_3
        tmp_2 = tmp_3 = None
        tmp_5 = torch._C._log_api_usage_once('python.nn_module')
        tmp_5 = None
        tmp_6 = torch.nn.functional.softmax(tmp_4, -1, _stacklevel=5)
        tmp_4 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = torch.matmul(tmp_7, in_3)
        tmp_7 = None
        tmp_9 = tmp_8.permute(0, 2, 1, 3)
        tmp_8 = None
        tmp_10 = tmp_9.contiguous()
        tmp_9 = None
        tmp_11 = tmp_10.view(2, 7, 768)
        tmp_10 = None
        return (tmp_11,)