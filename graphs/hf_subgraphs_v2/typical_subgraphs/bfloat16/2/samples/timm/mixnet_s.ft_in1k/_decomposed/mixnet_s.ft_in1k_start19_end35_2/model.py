import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        tmp_12 = torch.nn.functional.relu(in_12, inplace = True);  in_12 = None
        split = torch.functional.split(tmp_12, [48, 48], 1);  tmp_12 = None
        tmp_14 = split[0]
        tmp_15 = split[1];  split = None
        conv2d = torch.conv2d(tmp_14, in_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_4 = None
        conv2d_1 = torch.conv2d(tmp_15, in_5, None, (1, 1), (0, 0), (1, 1), 1);  tmp_15 = in_5 = None
        tmp_18 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_18 = in_0 = in_1 = in_3 = in_2 = None
        split_1 = torch.functional.split(tmp_19, [12, 12], 1)
        tmp_21 = split_1[0]
        tmp_22 = split_1[1];  split_1 = None
        conv2d_2 = torch.conv2d(tmp_21, in_10, None, (1, 1), (0, 0), (1, 1), 1);  tmp_21 = in_10 = None
        conv2d_3 = torch.conv2d(tmp_22, in_11, None, (1, 1), (0, 0), (1, 1), 1);  tmp_22 = in_11 = None
        tmp_25 = torch.cat([conv2d_2, conv2d_3], 1);  conv2d_2 = conv2d_3 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_25 = in_6 = in_7 = in_9 = in_8 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        return (tmp_19, tmp_27)
        