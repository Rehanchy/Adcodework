import oneflow
results = dict()
arg_0_tensor = oneflow.empty([2, 3, 4, 5], dtype=oneflow.float64)
arg_0_tensor.uniform_(-64, 0)
arg_0 = arg_0_tensor.clone().detach().to('cpu')
try:
    results['res_1'] = oneflow.DoubleTensor(arg_0)
except Exception as e:
    results['err_1'] = str(e)
arg_0 = arg_0_tensor.clone().detach().to('cpu')
try:
    results['res_2'] = oneflow.DoubleTensor(arg_0)
except Exception as e:
    results['err_2'] = str(e)

print(results)
