import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_6, inplace=True)
        tmp_5 = torch.conv2d(in_5, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_7 = tmp_6.view(2, 256, -1)
        tmp_6 = None
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 3)
        tmp_9 = tmp_8.view(2, 256, -1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 6)
        tmp_11 = tmp_10.view(2, 256, -1)
        tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 8)
        tmp_4 = None
        tmp_13 = tmp_12.view(2, 256, -1)
        tmp_12 = None
        tmp_14 = torch.cat([tmp_7, tmp_9, tmp_11, tmp_13], dim=2)
        tmp_7 = tmp_9 = tmp_11 = tmp_13 = None
        tmp_15 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
        tmp_16 = tmp_15.view(2, 256, -1)
        tmp_15 = None
        tmp_17 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 3)
        tmp_18 = tmp_17.view(2, 256, -1)
        tmp_17 = None
        tmp_19 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 6)
        tmp_20 = tmp_19.view(2, 256, -1)
        tmp_19 = None
        tmp_21 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 8)
        tmp_5 = None
        tmp_22 = tmp_21.view(2, 256, -1)
        tmp_21 = None
        tmp_23 = torch.cat([tmp_16, tmp_18, tmp_20, tmp_22], dim=2)
        tmp_16 = tmp_18 = tmp_20 = tmp_22 = None
        tmp_24 = tmp_14.reshape(2, 256, -1)
        tmp_14 = None
        tmp_25 = tmp_23.reshape(2, 256, -1)
        tmp_23 = None
        tmp_26 = tmp_25.permute(0, 2, 1)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = torch.matmul(in_4, tmp_24)
        tmp_24 = None
        tmp_29 = 0.0625 * tmp_28
        tmp_28 = None
        tmp_30 = torch.nn.functional.softmax(tmp_29, dim=-1)
        tmp_29 = None
        tmp_31 = torch.matmul(tmp_30, tmp_27)
        tmp_30 = tmp_27 = None
        tmp_32 = tmp_31.permute(0, 2, 1)
        tmp_31 = None
        tmp_33 = tmp_32.contiguous()
        tmp_32 = None
        tmp_34 = tmp_33.reshape(2, -1, 64, 128)
        tmp_33 = None
        tmp_35 = torch.conv2d(tmp_34, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_34 = tmp_1 = tmp_0 = None
        tmp_36 = torch.stack([tmp_35], dim=0)
        tmp_35 = None
        tmp_37 = tmp_36.sum(dim=0)
        tmp_36 = None
        tmp_38 = torch.cat([tmp_37, in_5], 1)
        tmp_37 = None
        return (tmp_38,)