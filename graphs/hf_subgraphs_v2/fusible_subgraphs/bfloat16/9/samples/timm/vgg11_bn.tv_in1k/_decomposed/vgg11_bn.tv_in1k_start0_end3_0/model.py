import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_6 = in_1 = in_0 = None
        tmp_8 = torch.nn.functional.batch_norm(conv2d, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  conv2d = in_2 = in_3 = in_5 = in_4 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = True);  tmp_8 = None
        return (tmp_9,)
        