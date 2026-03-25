import torch

from torch import device

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_1 : torch.Tensor):
        tmp_6 = in_0[0]
        tmp_7 = in_0[1]
        tmp_8 = tmp_6 != tmp_7;  tmp_6 = tmp_7 = None
        tmp_9 = torch.arange(0, 128, device = device(type='cuda'))
        tmp_10 = tmp_9.view(1, -1);  tmp_9 = None
        tmp_11 = tmp_10.repeat(2, 1);  tmp_10 = None
        tmp_12 = in_0[(slice(None, None, None), tmp_8)];  in_0 = tmp_8 = None
        sym_size_int = torch.ops.aten.sym_size.int(tmp_12, 1)
        _check_is_size_2 = torch._check_is_size(sym_size_int);  _check_is_size_2 = None
        ge_1 = sym_size_int >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1 = sym_size_int <= 128
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None
        _check_is_size_1 = torch._check_is_size(sym_size_int);  _check_is_size_1 = None
        tmp_20 = torch.cat([tmp_12, tmp_11], dim = 1);  tmp_12 = tmp_11 = None
        sym_sum = torch.sym_sum([128, sym_size_int]);  sym_size_int = None
        tmp_22 = torch.ones((sym_sum,), dtype = torch.float32, device = device(type='cuda'));  sym_sum = None
        tmp_23 = tmp_20[0]
        tmp_24 = tmp_20[1]
        tmp_25 = tmp_24.view((-1,))
        tmp_26 = tmp_25.expand_as(tmp_22);  tmp_25 = None
        tmp_27 = tmp_22.new_zeros((128,))
        tmp_28 = tmp_27.scatter_add_(0, tmp_26, tmp_22);  tmp_27 = tmp_26 = None
        tmp_29 = tmp_28.pow_(-0.5);  tmp_28 = None
        tmp_30 = tmp_29.__eq__(inf)
        tmp_31 = tmp_29.masked_fill_(tmp_30, 0);  tmp_30 = tmp_31 = None
        tmp_32 = tmp_29[tmp_23];  tmp_23 = None
        tmp_33 = tmp_32 * tmp_22;  tmp_32 = tmp_22 = None
        tmp_34 = tmp_29[tmp_24];  tmp_29 = tmp_24 = None
        tmp_35 = tmp_33 * tmp_34;  tmp_33 = tmp_34 = None
        linear = torch.nn.functional.linear(in_1, w_0, None);  in_1 = w_0 = None
        tmp_37 = tmp_20[1]
        tmp_38 = tmp_20[0];  tmp_20 = None
        tmp_39 = linear.index_select(-2, tmp_38);  linear = tmp_38 = None
        tmp_40 = tmp_35.view(-1, 1);  tmp_35 = None
        tmp_41 = tmp_40 * tmp_39;  tmp_40 = tmp_39 = None
        tmp_42 = tmp_37.view((-1, 1));  tmp_37 = None
        tmp_43 = tmp_42.expand_as(tmp_41);  tmp_42 = None
        tmp_44 = tmp_41.new_zeros((128, 128))
        tmp_45 = tmp_44.scatter_add_(0, tmp_43, tmp_41);  tmp_44 = tmp_43 = tmp_41 = None
        tmp_46 = tmp_45 + w_1;  tmp_45 = w_1 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, p = 0.0, training = False);  tmp_46 = None
        to = tmp_47.to(torch.float16);  tmp_47 = None
        linear_1 = torch.nn.functional.linear(to, w_3, w_2);  to = w_3 = w_2 = None
        return (linear_1,)
        