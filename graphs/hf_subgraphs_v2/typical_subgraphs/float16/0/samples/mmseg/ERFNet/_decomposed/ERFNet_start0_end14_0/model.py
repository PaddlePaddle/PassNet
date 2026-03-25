import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_6, in_5, (2, 2), (1, 1), (1, 1), 1);  in_6 = in_5 = None
        tmp_16 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  in_0 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, (256, 256), None, 'bilinear', False);  tmp_16 = None
        tmp_18 = torch.cat([conv2d, tmp_17], 1);  conv2d = tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_18 = in_1 = in_2 = in_4 = in_3 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace = False);  tmp_19 = None
        conv2d_1 = torch.conv2d(tmp_20, in_12, in_11, (2, 2), (1, 1), (1, 1), 1);  in_12 = in_11 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_20, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_20 = None
        tmp_23 = torch.nn.functional.interpolate(tmp_22, (128, 128), None, 'bilinear', False);  tmp_22 = None
        tmp_24 = torch.cat([conv2d_1, tmp_23], 1);  conv2d_1 = tmp_23 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, in_7, in_8, in_10, in_9, False, 0.1, 0.001);  tmp_24 = in_7 = in_8 = in_10 = in_9 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace = False);  tmp_25 = None
        conv2d_2 = torch.conv2d(tmp_26, in_14, in_13, (1, 1), (1, 0), (1, 1), 1);  in_14 = in_13 = None
        tmp_28 = torch.nn.functional.relu(conv2d_2, inplace = False);  conv2d_2 = None
        return (tmp_26, tmp_28)
        