import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_7 = torch.nn.functional.relu(in_7, inplace = False);  in_7 = None
        tmp_8 = in_6 * tmp_7;  in_6 = tmp_7 = None
        tmp_9 = tmp_8 + in_5;  tmp_8 = in_5 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1);  tmp_9 = None
        conv2d = torch.conv2d(tmp_10, in_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_2 = None
        tmp_12 = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
        tmp_13 = in_4 * tmp_12;  in_4 = tmp_12 = None
        tmp_14 = tmp_13 + in_3;  tmp_13 = in_3 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = tmp_15.flatten(1, -1);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, in_1, in_0);  tmp_16 = in_1 = in_0 = None
        return (linear,)
        