import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.relu(in_11, inplace = True);  in_11 = None
        tmp_11 = torch.flatten(tmp_10, 2);  tmp_10 = None
        tmp_12 = torch.functional.norm(tmp_11, dim = -1, keepdim = True)
        tmp_13 = tmp_12 * 0.07216878364870322;  tmp_12 = None
        tmp_14 = tmp_13.clamp(min = 1e-05);  tmp_13 = None
        tmp_15 = tmp_11 / tmp_14;  tmp_11 = tmp_14 = None
        tmp_16 = tmp_15 * in_8;  tmp_15 = in_8 = None
        linear = torch.nn.functional.linear(tmp_16, in_9, None);  tmp_16 = in_9 = None
        tmp_18 = torch.cat([in_10, linear], dim = 2);  in_10 = linear = None
        tmp_19 = torch.functional.norm(tmp_18, dim = -1, keepdim = True)
        tmp_20 = tmp_19 * 0.0625;  tmp_19 = None
        tmp_21 = tmp_20.clamp(min = 1e-05);  tmp_20 = None
        tmp_22 = tmp_18 / tmp_21;  tmp_21 = None
        tmp_23 = tmp_22 * in_2;  tmp_22 = in_2 = None
        linear_1 = torch.nn.functional.linear(tmp_23, in_5, None);  tmp_23 = in_5 = None
        tmp_25 = torch.nn.functional.silu(linear_1, inplace = True);  linear_1 = None
        split = torch.functional.split(tmp_25, [512, 512, 128], dim = 2);  tmp_25 = None
        tmp_27 = split[0]
        tmp_28 = split[1]
        tmp_29 = split[2];  split = None
        tmp_30 = tmp_29.unsqueeze(2);  tmp_29 = None
        tmp_31 = in_7[(None, None, slice(None, None, None))];  in_7 = None
        tmp_32 = tmp_30 * tmp_31;  tmp_30 = tmp_31 = None
        tmp_33 = tmp_32 + in_6;  tmp_32 = in_6 = None
        unbind = torch.unbind(tmp_33, dim = 2);  tmp_33 = None
        tmp_35 = unbind[0]
        tmp_36 = unbind[1];  unbind = None
        tmp_37 = tmp_36.permute(0, 2, 1);  tmp_36 = None
        bmm = torch.bmm(tmp_35, tmp_37);  tmp_35 = tmp_37 = None
        tmp_39 = bmm / 11.313708498984761;  bmm = None
        tmp_40 = torch.nn.functional.relu(tmp_39);  tmp_39 = None
        tmp_41 = torch.square(tmp_40);  tmp_40 = None
        bmm_1 = torch.bmm(tmp_41, tmp_28);  tmp_41 = tmp_28 = None
        tmp_43 = tmp_27 * bmm_1;  tmp_27 = bmm_1 = None
        linear_2 = torch.nn.functional.linear(tmp_43, in_3, None);  tmp_43 = in_3 = None
        tmp_45 = tmp_18 * in_4;  tmp_18 = in_4 = None
        tmp_46 = tmp_45 + linear_2;  tmp_45 = linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_46, in_0, None);  in_0 = None
        linear_4 = torch.nn.functional.linear(tmp_46, in_1, None);  tmp_46 = in_1 = None
        return (linear_3, linear_4)
        