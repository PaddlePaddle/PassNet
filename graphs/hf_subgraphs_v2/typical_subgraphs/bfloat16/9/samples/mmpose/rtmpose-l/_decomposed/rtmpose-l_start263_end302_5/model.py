import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, in_0 : torch.Tensor):
        tmp_12 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_12, w_3, w_2, (1, 1), (3, 3), (1, 1), 1);  tmp_12 = w_3 = w_2 = None
        tmp_14 = torch.flatten(conv2d, 2);  conv2d = None
        tmp_15 = torch.functional.norm(tmp_14, dim = -1, keepdim = True)
        tmp_16 = tmp_15 * 0.14433756729740643;  tmp_15 = None
        tmp_17 = tmp_16.clamp(min = 1e-05);  tmp_16 = None
        tmp_18 = tmp_14 / tmp_17;  tmp_14 = tmp_17 = None
        tmp_19 = tmp_18 * w_10;  tmp_18 = w_10 = None
        linear = torch.nn.functional.linear(tmp_19, w_11, None);  tmp_19 = w_11 = None
        tmp_21 = torch.functional.norm(linear, dim = -1, keepdim = True)
        tmp_22 = tmp_21 * 0.0625;  tmp_21 = None
        tmp_23 = tmp_22.clamp(min = 1e-05);  tmp_22 = None
        tmp_24 = linear / tmp_23;  tmp_23 = None
        tmp_25 = tmp_24 * w_4;  tmp_24 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_25, w_7, None);  tmp_25 = w_7 = None
        tmp_27 = torch.nn.functional.silu(linear_1, inplace = True);  linear_1 = None
        split = torch.functional.split(tmp_27, [512, 512, 128], dim = 2);  tmp_27 = None
        tmp_29 = split[0]
        tmp_30 = split[1]
        tmp_31 = split[2];  split = None
        tmp_32 = tmp_31.unsqueeze(2);  tmp_31 = None
        tmp_33 = w_9[(None, None, slice(None, None, None))];  w_9 = None
        tmp_34 = tmp_32 * tmp_33;  tmp_32 = tmp_33 = None
        tmp_35 = tmp_34 + w_8;  tmp_34 = w_8 = None
        unbind = torch.unbind(tmp_35, dim = 2);  tmp_35 = None
        tmp_37 = unbind[0]
        tmp_38 = unbind[1];  unbind = None
        tmp_39 = tmp_38.permute(0, 2, 1);  tmp_38 = None
        bmm = torch.bmm(tmp_37, tmp_39);  tmp_37 = tmp_39 = None
        tmp_41 = bmm / 11.313708498984761;  bmm = None
        tmp_42 = torch.nn.functional.relu(tmp_41);  tmp_41 = None
        tmp_43 = torch.square(tmp_42);  tmp_42 = None
        bmm_1 = torch.bmm(tmp_43, tmp_30);  tmp_43 = tmp_30 = None
        tmp_45 = tmp_29 * bmm_1;  tmp_29 = bmm_1 = None
        linear_2 = torch.nn.functional.linear(tmp_45, w_5, None);  tmp_45 = w_5 = None
        tmp_47 = linear * w_6;  linear = w_6 = None
        tmp_48 = tmp_47 + linear_2;  tmp_47 = linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_48, w_0, None);  w_0 = None
        linear_4 = torch.nn.functional.linear(tmp_48, w_1, None);  tmp_48 = w_1 = None
        return (linear_3, linear_4)
        