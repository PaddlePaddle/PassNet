import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_4, in_1, in_0);  in_4 = in_1 = in_0 = None
        tmp_4 = linear.reshape(1, 257, 16, -1);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_3[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_7 = in_3[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  in_3 = None
        tensor_split = in_2.tensor_split(2, -1);  in_2 = None
        tmp_9 = tensor_split[0]
        tmp_10 = tensor_split[1];  tensor_split = None
        return (tmp_10, tmp_6, tmp_7, tmp_9, tmp_5)
        