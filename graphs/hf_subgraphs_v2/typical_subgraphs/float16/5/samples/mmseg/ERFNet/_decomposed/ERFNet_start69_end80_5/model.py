import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.dropout(in_9, 0.1, False, False);  in_9 = None
        tmp_9 = tmp_8 + in_8;  tmp_8 = in_8 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = False);  tmp_9 = None
        conv2d = torch.conv2d(tmp_10, in_5, in_4, (2, 2), (1, 1), (1, 1), 1);  in_5 = in_4 = None
        tmp_12 = torch.nn.functional.max_pool2d(tmp_10, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_10 = None
        tmp_13 = torch.nn.functional.interpolate(tmp_12, (64, 64), None, 'bilinear', False);  tmp_12 = None
        tmp_14 = torch.cat([conv2d, tmp_13], 1);  conv2d = tmp_13 = None
        tmp_15 = torch.nn.functional.batch_norm(tmp_14, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_14 = in_0 = in_1 = in_3 = in_2 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace = False);  tmp_15 = None
        conv2d_1 = torch.conv2d(tmp_16, in_7, in_6, (1, 1), (1, 0), (1, 1), 1);  in_7 = in_6 = None
        tmp_18 = torch.nn.functional.relu(conv2d_1, inplace = False);  conv2d_1 = None
        return (tmp_16, tmp_18)
        