import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2):
        tmp_0 = torch.nn.functional.linear(in_0, w_2, w_1)
        tmp_1 = tmp_0.reshape(1, 197, 12, -1)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = in_1[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_4 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_5 = w_0.tensor_split(2, -1)
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_5 = None
        return (tmp_2, tmp_3, tmp_4, tmp_6, tmp_7)