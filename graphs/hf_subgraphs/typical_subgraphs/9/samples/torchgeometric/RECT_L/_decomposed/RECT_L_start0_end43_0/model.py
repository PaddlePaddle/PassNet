import torch
from torch import device
from torch import inf

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = in_1
        tmp_6 = tmp_0[0]
        tmp_7 = tmp_0[1]
        tmp_8 = tmp_6 != tmp_7
        tmp_6 = tmp_7 = None
        tmp_9 = torch.arange(0, 128, device=device(type='cuda'))
        tmp_10 = tmp_9.view(1, -1)
        tmp_9 = None
        tmp_11 = tmp_10.repeat(2, 1)
        tmp_10 = None
        tmp_12 = tmp_0[slice(None, None, None), tmp_8]
        tmp_0 = tmp_8 = None
        tmp_13 = torch.ops.aten.sym_size.int(tmp_12, 1)
        tmp_14 = torch._check_is_size(tmp_13)
        tmp_14 = None
        tmp_15 = tmp_13 >= 0
        tmp_16 = torch.ops.aten._assert_scalar.default(tmp_15, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_13 <= 128
        tmp_18 = torch.ops.aten._assert_scalar.default(tmp_17, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'")
        tmp_17 = tmp_18 = None
        tmp_19 = torch._check_is_size(tmp_13)
        tmp_19 = None
        tmp_20 = torch.cat([tmp_12, tmp_11], dim=1)
        tmp_12 = tmp_11 = None
        tmp_21 = torch.sym_sum([128, tmp_13])
        tmp_13 = None
        tmp_22 = torch.ones((tmp_21,), dtype=torch.float32, device=device(type='cuda'))
        tmp_21 = None
        tmp_23 = tmp_20[0]
        tmp_24 = tmp_20[1]
        tmp_25 = tmp_24.view((-1,))
        tmp_26 = tmp_25.expand_as(tmp_22)
        tmp_25 = None
        tmp_27 = tmp_22.new_zeros((128,))
        tmp_28 = tmp_27.scatter_add_(0, tmp_26, tmp_22)
        tmp_27 = tmp_26 = None
        tmp_29 = tmp_28.pow_(-0.5)
        tmp_28 = None
        tmp_30 = tmp_29.__eq__(inf)
        tmp_31 = tmp_29.masked_fill_(tmp_30, 0)
        tmp_30 = tmp_31 = None
        tmp_32 = tmp_29[tmp_23]
        tmp_23 = None
        tmp_33 = tmp_32 * tmp_22
        tmp_32 = tmp_22 = None
        tmp_34 = tmp_29[tmp_24]
        tmp_29 = tmp_24 = None
        tmp_35 = tmp_33 * tmp_34
        tmp_33 = tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_5, tmp_1, None)
        tmp_5 = tmp_1 = None
        tmp_37 = tmp_20[1]
        tmp_38 = tmp_20[0]
        tmp_20 = None
        tmp_39 = tmp_36.index_select(-2, tmp_38)
        tmp_36 = tmp_38 = None
        tmp_40 = tmp_35.view(-1, 1)
        tmp_35 = None
        tmp_41 = tmp_40 * tmp_39
        tmp_40 = tmp_39 = None
        tmp_42 = tmp_37.view((-1, 1))
        tmp_37 = None
        tmp_43 = tmp_42.expand_as(tmp_41)
        tmp_42 = None
        tmp_44 = tmp_41.new_zeros((128, 128))
        tmp_45 = tmp_44.scatter_add_(0, tmp_43, tmp_41)
        tmp_44 = tmp_43 = tmp_41 = None
        tmp_46 = tmp_45 + tmp_2
        tmp_45 = tmp_2 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, p=0.0, training=False)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_47, tmp_4, tmp_3)
        tmp_47 = tmp_4 = tmp_3 = None
        return (tmp_48,)