import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_11, w_10, (4, 4), (3, 3), (1, 1), 1);  in_0 = w_11 = w_10 = None
        tmp_14 = conv2d.flatten(2);  conv2d = None
        tmp_15 = tmp_14.transpose(1, 2);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (16,), w_9, w_8, 1e-05);  tmp_15 = w_9 = w_8 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (16,), w_7, w_6, 1e-05);  w_7 = w_6 = None
        linear = torch.nn.functional.linear(tmp_17, w_3, w_2);  w_3 = w_2 = None
        tmp_19 = linear.view(1, -1, 1, 16);  linear = None
        tmp_20 = tmp_19.transpose(1, 2);  tmp_19 = None
        tmp_21 = tmp_17.permute(0, 2, 1);  tmp_17 = None
        tmp_22 = tmp_21.reshape(1, 16, 16, 16);  tmp_21 = None
        conv2d_1 = torch.conv2d(tmp_22, w_5, w_4, (8, 8), (0, 0), (1, 1), 1);  tmp_22 = w_5 = w_4 = None
        tmp_24 = conv2d_1.reshape(1, 16, -1);  conv2d_1 = None
        tmp_25 = tmp_24.permute(0, 2, 1);  tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (16,), w_1, w_0, 1e-05);  tmp_25 = w_1 = w_0 = None
        return (tmp_16, tmp_26, tmp_20)
        