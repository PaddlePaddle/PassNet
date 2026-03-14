import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_2.view(1, 18, 16, 3, 96)
        tmp_1 = tmp_0[Ellipsis, 0, slice(None, None, None)]
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_0[Ellipsis, 1, slice(None, None, None)]
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_0[Ellipsis, 2, slice(None, None, None)]
        tmp_0 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_2.reshape(16, -1, 96)
        tmp_2 = None
        tmp_8 = tmp_4.reshape(16, -1, 96)
        tmp_4 = None
        tmp_9 = tmp_8.transpose(-1, -2)
        tmp_8 = None
        tmp_10 = tmp_6.reshape(16, -1, 96)
        tmp_6 = None
        tmp_11 = in_0.baddbmm(batch1=tmp_7, batch2=tmp_9, beta=1.0, alpha=0.10206207261596577)
        tmp_7 = tmp_9 = None
        tmp_12 = tmp_11.view(1, 16, 18, -1)
        tmp_11 = None
        tmp_13 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)]
        tmp_14 = tmp_12 + tmp_13
        tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim=-1, dtype=torch.float32)
        tmp_14 = None
        tmp_16 = tmp_15.to(torch.float16)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = tmp_17.view(16, 18, -1)
        tmp_17 = None
        tmp_19 = torch.bmm(tmp_18, tmp_10)
        tmp_18 = tmp_10 = None
        tmp_20 = tmp_19.view(1, 16, 18, 96)
        tmp_19 = None
        tmp_21 = tmp_20.permute(0, 2, 1, 3)
        tmp_20 = None
        tmp_22 = tmp_21.reshape(1, 18, 1536)
        tmp_21 = None
        return (tmp_22,)