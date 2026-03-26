import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, in_0 : torch.Tensor):
        tmp_3 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, 1);  tmp_3 = None
        conv2d = torch.conv2d(tmp_4, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_2 = None
        tmp_6 = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        tmp_8 = tmp_7.flatten(1, -1);  tmp_7 = None
        linear = torch.nn.functional.linear(tmp_8, w_1, w_0);  tmp_8 = w_1 = w_0 = None
        return (linear,)
        