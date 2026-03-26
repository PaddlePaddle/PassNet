import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_12, in_11, (4, 4), (3, 3), (1, 1), 1);  in_0 = in_12 = in_11 = None
        tmp_14 = conv2d.flatten(2);  conv2d = None
        tmp_15 = tmp_14.transpose(1, 2);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (32,), in_10, in_9, 1e-05);  tmp_15 = in_10 = in_9 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_8, in_7, 1e-05);  in_8 = in_7 = None
        linear = torch.nn.functional.linear(tmp_17, in_4, in_3);  in_4 = in_3 = None
        tmp_19 = linear.view(32, -1, 1, 32);  linear = None
        tmp_20 = tmp_19.transpose(1, 2);  tmp_19 = None
        tmp_21 = tmp_17.permute(0, 2, 1);  tmp_17 = None
        tmp_22 = tmp_21.reshape(32, 32, 128, 128);  tmp_21 = None
        conv2d_1 = torch.conv2d(tmp_22, in_6, in_5, (8, 8), (0, 0), (1, 1), 1);  tmp_22 = in_6 = in_5 = None
        tmp_24 = conv2d_1.reshape(32, 32, -1);  conv2d_1 = None
        tmp_25 = tmp_24.permute(0, 2, 1);  tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (32,), in_2, in_1, 1e-05);  tmp_25 = in_2 = in_1 = None
        return (tmp_16, tmp_26, tmp_20)
        