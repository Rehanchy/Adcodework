import oneflow

def restore_grad(args):
    if type(args).__name__ == 'tuple':
        temp_list_reverse = []
        for arg in args:
            if type(arg).__name__ == 'Parameter' or type(arg).__name__ == 'Tensor':
                arg = arg.clone().detach().requires_grad_(True)
            temp_list_reverse.append(arg)
        return tuple(temp_list_reverse)
    else:
        temp_dict_reverse = {}
        for item in args.items():
            arg = item[1]
            if type(item[1]).__name__ == 'Parameter' or type(item[1]).__name__ == 'Tensor':
                arg = item[1].clone().detach().requires_grad_(True)
            temp_dict_reverse[item[0]] = arg
        return temp_dict_reverse

def restore_cuda(args):
    if type(args).__name__ == 'tuple':
        temp_list = []
        for arg in args:
            if type(arg).__name__ == 'Parameter' or type(arg).__name__ == 'Tensor':
                arg = arg.clone().cuda()
            temp_list.append(arg)
        return tuple(temp_list)
    else:
        temp_dict = {}
        for item in args.items():
            arg = item[1]
            if type(item[1]).__name__ == 'Parameter' or type(item[1]).__name__ == 'Tensor':
                arg = item[1].clone().cuda()
            temp_dict[item[0]] = arg
        return temp_dict

path = '/mnt/sda/yjy/summer/DL-autograd-torch-main/oneflow_error_input_value/689_99'

[api_name, arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct] = oneflow.load(path)

arg_fuzzed_reverse=restore_grad(arg_fuzzed_direct)
kwarg_fuzzed_reverse=restore_grad(kwarg_fuzzed_direct)
input_arg_fuzzed_reverse=restore_grad(input_arg_fuzzed_direct)
input_kwarg_fuzzed_reverse=restore_grad(input_kwarg_fuzzed_direct)

arg_fuzzed_reverse=restore_cuda(arg_fuzzed_direct)
kwarg_fuzzed_reverse=restore_cuda(kwarg_fuzzed_direct)
input_arg_fuzzed_reverse=restore_cuda(input_arg_fuzzed_direct)
input_kwarg_fuzzed_reverse=restore_cuda(input_kwarg_fuzzed_direct)
arg_fuzzed_direct=restore_cuda(arg_fuzzed_direct)
kwarg_fuzzed_direct=restore_cuda(kwarg_fuzzed_direct)
input_arg_fuzzed_direct=restore_cuda(input_arg_fuzzed_direct)
input_kwarg_fuzzed_direct=restore_cuda(input_kwarg_fuzzed_direct)

api = eval(api_name)

if type(api).__name__ == 'type':
        api_obj = api(*input_arg_fuzzed_reverse,**input_kwarg_fuzzed_reverse)
        output_reverse = api_obj(*arg_fuzzed_reverse,**kwarg_fuzzed_reverse)
else:
    output_reverse = api(*arg_fuzzed_reverse,**kwarg_fuzzed_reverse)

if type(api).__name__ == 'type':
        api_obj = api(*input_arg_fuzzed_direct,**input_kwarg_fuzzed_direct)
        output_direct = api_obj(*arg_fuzzed_direct,**kwarg_fuzzed_direct)
else:
    output_direct = api(*arg_fuzzed_direct,**kwarg_fuzzed_direct)



#print((arg_fuzzed_direct[0]==arg_fuzzed_reverse[0]).all())
print(input_arg_fuzzed_direct)
print(arg_fuzzed_reverse[0].dtype)
'''
print(kwarg_fuzzed_reverse)
print(arg_fuzzed_reverse[0].requires_grad)
'''

#print(oneflow.sum(output_direct))
#print(oneflow.sum(output_reverse))

print(oneflow.sum(output_reverse)==oneflow.sum(output_direct))
#print((output_reverse[0]==output_direct[0]).all())
#print(output_reverse)
