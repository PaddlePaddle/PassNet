import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_9, inplace=True)
        tmp_5 = tmp_4.view(1, 512, 64, 64)
        tmp_4 = None
        tmp_6 = in_4.view(512, 1, 7, 7)
        tmp_7 = torch.nn.functional.pad(tmp_5, (3, 3, 3, 3), 'constant', 0)
        tmp_5 = None
        tmp_8 = torch.conv2d(input=tmp_7, weight=tmp_6, groups=512)
        tmp_7 = tmp_6 = None
        tmp_9 = tmp_8.view(1, 512, 64, 64)
        tmp_8 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_9 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace=False)
        tmp_10 = None
        tmp_12 = torch.cat([in_5, in_7, in_8, in_6, tmp_11], dim=1)
        tmp_11 = None
        return (tmp_12,)