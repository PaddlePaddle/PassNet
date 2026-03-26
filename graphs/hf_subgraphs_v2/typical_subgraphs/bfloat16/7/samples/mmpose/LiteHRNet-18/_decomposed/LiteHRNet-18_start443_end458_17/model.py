import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        in_0 += in_1;  in_5 = in_0;  in_0 = in_1 = None
        in_5 += in_4;  tmp_0 = in_5;  in_5 = in_4 = None
        tmp_2 = torch.nn.functional.relu(tmp_0, inplace = False);  tmp_0 = None
        chunk = in_2.chunk(2, dim = 1);  in_2 = None
        tmp_4 = chunk[0]
        tmp_5 = chunk[1];  chunk = None
        chunk_1 = in_3.chunk(2, dim = 1);  in_3 = None
        tmp_7 = chunk_1[0]
        tmp_8 = chunk_1[1];  chunk_1 = None
        chunk_2 = tmp_2.chunk(2, dim = 1);  tmp_2 = None
        tmp_10 = chunk_2[0]
        tmp_11 = chunk_2[1];  chunk_2 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, (16, 12))
        tmp_13 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, (16, 12))
        tmp_14 = torch.cat([tmp_12, tmp_13, tmp_11], dim = 1);  tmp_12 = tmp_13 = None
        return (tmp_14, tmp_4, tmp_7, tmp_10, tmp_5, tmp_8, tmp_11)
        