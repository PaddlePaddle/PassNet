import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, in_0 : torch.Tensor):
        tmp_7 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_8 = w_6 * tmp_7;  w_6 = tmp_7 = None
        tmp_9 = tmp_8 + w_5;  tmp_8 = w_5 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1);  tmp_9 = None
        conv2d = torch.conv2d(tmp_10, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = w_2 = None
        tmp_12 = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
        tmp_13 = w_4 * tmp_12;  w_4 = tmp_12 = None
        tmp_14 = tmp_13 + w_3;  tmp_13 = w_3 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, w_1, w_0);  tmp_16 = w_1 = w_0 = None
        return (linear,)
        