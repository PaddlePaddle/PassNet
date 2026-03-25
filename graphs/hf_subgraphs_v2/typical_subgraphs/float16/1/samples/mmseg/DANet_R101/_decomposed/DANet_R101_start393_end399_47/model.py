import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.relu(in_4, inplace = True);  in_4 = None
        tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
        conv2d = torch.conv2d(tmp_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_1 = in_0 = None
        tmp_7 = in_5 + tmp_4;  in_5 = tmp_4 = None
        tmp_8 = torch.nn.functional.dropout2d(tmp_7, 0.1, False, False);  tmp_7 = None
        conv2d_1 = torch.conv2d(tmp_8, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_3 = in_2 = None
        return (conv2d, conv2d_1)
        