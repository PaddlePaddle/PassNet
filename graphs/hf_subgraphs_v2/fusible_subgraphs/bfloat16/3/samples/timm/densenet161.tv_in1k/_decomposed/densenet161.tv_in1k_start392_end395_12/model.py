import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40):
        tmp_4 = torch.cat([in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40], 1);  in_4 = in_5 = in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = in_17 = in_18 = in_19 = in_20 = in_21 = in_22 = in_23 = in_24 = in_25 = in_26 = in_27 = in_28 = in_29 = in_30 = in_31 = in_32 = in_33 = in_34 = in_35 = in_36 = in_37 = in_38 = in_39 = in_40 = None
        tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_4 = in_0 = in_1 = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace = True);  tmp_5 = None
        return (tmp_6,)
        