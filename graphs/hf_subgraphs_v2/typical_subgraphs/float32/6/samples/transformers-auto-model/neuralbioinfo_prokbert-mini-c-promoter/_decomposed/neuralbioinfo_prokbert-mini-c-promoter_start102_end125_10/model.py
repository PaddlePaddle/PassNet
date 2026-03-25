import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_2.transpose(-1, -2)
        tmp_2 = torch.matmul(in_3, tmp_1)
        tmp_1 = None
        tmp_3 = torch.arange(1024, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_4 = tmp_3.view(-1, 1)
        tmp_3 = None
        tmp_5 = torch.arange(1024, dtype=torch.int64, device=device(type='cuda', index=0))
        tmp_6 = tmp_5.view(1, -1)
        tmp_5 = None
        tmp_7 = tmp_4 - tmp_6
        tmp_4 = tmp_6 = None
        tmp_8 = tmp_7 + 2048
        tmp_7 = None
        tmp_9 = tmp_8 - 1
        tmp_8 = None
        tmp_10 = torch.nn.functional.embedding(tmp_9, tmp_0, None, None, 2.0, False, False)
        tmp_9 = tmp_0 = None
        tmp_11 = tmp_10.to(dtype=torch.float32)
        tmp_10 = None
        tmp_12 = torch.functional.einsum('bhld,lrd->bhlr', in_3, tmp_11)
        tmp_13 = torch.functional.einsum('bhrd,lrd->bhlr', in_2, tmp_11)
        tmp_11 = None
        tmp_14 = tmp_2 + tmp_12
        tmp_2 = tmp_12 = None
        tmp_15 = tmp_14 + tmp_13
        tmp_14 = tmp_13 = None
        tmp_16 = tmp_15 / 8.0
        tmp_15 = None
        tmp_17 = tmp_16 + in_1
        tmp_16 = None
        tmp_18 = torch.nn.functional.softmax(tmp_17, dim=-1)
        tmp_17 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.1, False, False)
        tmp_18 = None
        tmp_20 = torch.matmul(tmp_19, in_4)
        tmp_19 = None
        tmp_21 = tmp_20.permute(0, 2, 1, 3)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = tmp_22.view((2, 1024, 384))
        tmp_22 = None
        return (tmp_23,)