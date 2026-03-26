class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [24, 1, 16384, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.063
	std = 0.395


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [24, 1, 64, 256]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.115
	std = 0.163


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [24, 1, 256, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.004
	std = 0.029
