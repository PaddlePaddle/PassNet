import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_7[slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None)]
        tmp_5 = in_6 + in_5
        tmp_6 = torch.cat([in_4, tmp_4], dim=1)
        tmp_4 = None
        tmp_7 = torch.cat((tmp_5, tmp_6), dim=1)
        tmp_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_7 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
        tmp_8 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
        tmp_9 = None
        return (tmp_10,)