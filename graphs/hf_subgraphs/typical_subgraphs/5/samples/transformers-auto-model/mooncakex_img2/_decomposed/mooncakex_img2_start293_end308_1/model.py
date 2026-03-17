import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(2, -1, 12, 64)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_4.transpose(-1, -2)
        tmp_6 = torch.matmul(in_5, tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6 / 8.0
        tmp_6 = None
        tmp_8 = in_3.to(device(type='cuda', index=0))
        tmp_9 = tmp_7 + tmp_8
        tmp_7 = tmp_8 = None
        tmp_10 = torch._C._log_api_usage_once('python.nn_module')
        tmp_10 = None
        tmp_11 = torch.nn.functional.softmax(tmp_9, -1, _stacklevel=5)
        tmp_9 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = torch.matmul(tmp_12, tmp_4)
        tmp_12 = tmp_4 = None
        tmp_14 = tmp_13.permute(0, 2, 1, 3)
        tmp_13 = None
        tmp_15 = tmp_14.contiguous()
        tmp_14 = None
        tmp_16 = tmp_15.view(2, 7, 768)
        tmp_15 = None
        return (tmp_16,)