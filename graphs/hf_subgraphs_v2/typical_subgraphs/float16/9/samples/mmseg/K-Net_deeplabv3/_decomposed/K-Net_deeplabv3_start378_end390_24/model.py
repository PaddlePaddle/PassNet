import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = torch.nn.functional.layer_norm(in_0, (512,), w_3, w_2, 1e-05);  in_0 = w_3 = w_2 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
        linear = torch.nn.functional.linear(tmp_5, w_1, w_0);  tmp_5 = w_1 = w_0 = None
        tmp_7 = linear.permute(0, 1, 3, 2);  linear = None
        tmp_8 = tmp_7.reshape(1, 150, 512, 1, 1);  tmp_7 = None
        tmp_9 = in_2[slice(0, 1, None)];  in_2 = None
        tmp_10 = tmp_8[0];  tmp_8 = None
        conv2d = torch.conv2d(tmp_9, tmp_10, padding = 0);  tmp_9 = tmp_10 = None
        tmp_12 = torch.cat([conv2d], dim = 0);  conv2d = None
        tmp_13 = tmp_12.reshape(1, 150, 64, 64);  tmp_12 = None
        tmp_14 = in_1.permute(0, 1, 3, 2);  in_1 = None
        tmp_15 = tmp_14.reshape(1, 150, 512, 1, 1);  tmp_14 = tmp_15 = None
        return (tmp_13,)
        