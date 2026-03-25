import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_5, w_4, (2, 2), (1, 1), (1, 1), 1);  w_5 = w_4 = None
        tmp_16 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  in_0 = None
        tmp_17 = torch.nn.functional.interpolate(tmp_16, (256, 256), None, 'bilinear', False);  tmp_16 = None
        tmp_18 = torch.cat([conv2d, tmp_17], 1);  conv2d = tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, w_0, w_1, w_3, w_2, False, 0.1, 0.001);  tmp_18 = w_0 = w_1 = w_3 = w_2 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace = False);  tmp_19 = None
        conv2d_1 = torch.conv2d(tmp_20, w_11, w_10, (2, 2), (1, 1), (1, 1), 1);  w_11 = w_10 = None
        tmp_22 = torch.nn.functional.max_pool2d(tmp_20, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  tmp_20 = None
        tmp_23 = torch.nn.functional.interpolate(tmp_22, (128, 128), None, 'bilinear', False);  tmp_22 = None
        tmp_24 = torch.cat([conv2d_1, tmp_23], 1);  conv2d_1 = tmp_23 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, w_6, w_7, w_9, w_8, False, 0.1, 0.001);  tmp_24 = w_6 = w_7 = w_9 = w_8 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace = False);  tmp_25 = None
        conv2d_2 = torch.conv2d(tmp_26, w_13, w_12, (1, 1), (1, 0), (1, 1), 1);  w_13 = w_12 = None
        tmp_28 = torch.nn.functional.relu(conv2d_2, inplace = False);  conv2d_2 = None
        return (tmp_26, tmp_28)
        