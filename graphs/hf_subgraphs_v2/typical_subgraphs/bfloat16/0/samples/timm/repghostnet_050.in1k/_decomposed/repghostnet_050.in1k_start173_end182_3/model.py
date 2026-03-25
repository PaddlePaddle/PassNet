import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_8 = torch.nn.functional.batch_norm(in_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_8 = in_0 = in_1 = in_3 = in_2 = None
        tmp_9 = in_9 + tmp_8;  in_9 = tmp_8 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = False);  tmp_9 = None
        tmp_11 = tmp_10.mean((2, 3), keepdim = True)
        conv2d = torch.conv2d(tmp_11, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = in_7 = in_6 = None
        tmp_13 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_13, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = in_5 = in_4 = None
        tmp_15 = torch.nn.functional.hardsigmoid(conv2d_1, False);  conv2d_1 = None
        tmp_16 = tmp_10 * tmp_15;  tmp_10 = tmp_15 = None
        return (tmp_16,)
        