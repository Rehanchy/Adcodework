import oneflow
import pymongo
import random
import numpy as np
from sympy import arg


def get_info():
    host = "localhost"
    port = 27017
    mongo_db = "OneflowDatabase"
    client = pymongo.MongoClient(host=host, port=port)
    db = client[mongo_db]
    #random_collection_name = random.sample(db.list_collection_names(),1)[0]
    random_collection_name = 'oneflow.nn.TripletMarginLoss'
    #random_collection_name = 'oneflow.nn.CrossEntropyLoss'
    print(random_collection_name)
    print(type(random_collection_name))
    random_collection = db[random_collection_name]
    random_input = random.sample(list(random_collection.find({})), 1)[0]
    #random_input = random_collection.find({})[1]
    print(random_input)
    return random_collection_name, random_input
    '''
    collection = db[collection_name]
    x = collection.find_one({})
    print(x)
    '''
def handle_params(params):
    del params['_id']
    arg_list = []
    kwarg_dict = {}
    input_arg_list = []
    input_kwarg_dict = {}
    for item in params.items():
        if item[0].startswith('init_'):
            if item[0].startswith('init_parameter_'):
                input_arg_list.append(item[1])
            else:
                item_name = item[0].replace('init_','')
                input_kwarg_dict[item_name] = item[1]

        else:
            if item[0].startswith('parameter_'):
                arg_list.append(item[1])
            else:
                kwarg_dict[item[0]] = item[1]
    return arg_list, kwarg_dict, input_arg_list, input_kwarg_dict



def restore_or_generate(arg, mode="reverse",dtype_for_value_check=oneflow.float64):
    arg_type = type(arg).__name__
    if arg_type == 'dict':
        if 'type' in arg.keys():
            if arg['type'] == 'Tensor':
                if arg['dtype'] =='oneflow.bool':
                    t = oneflow.randint(0,2,tuple(arg['shape']))
                    t = t.to(oneflow.bool)
                elif arg['dtype'].startswith('oneflow.int'):
                    t = oneflow.randint(0,5,tuple(arg['shape']))
                else:
                    #t = oneflow.rand(*tuple(arg['shape']),dtype= eval(arg['dtype']), requires_grad=True)
                    if mode == 'reverse':
                        t = oneflow.rand(*tuple(arg['shape']),dtype= oneflow.float64, requires_grad=True)
                    elif mode == 'reverse_for_value_check':
                        t = oneflow.rand(*tuple(arg['shape']),dtype= dtype_for_value_check, requires_grad=True)
                    t.uniform_(-5,5)
                return t
            elif arg['type'] == 'Parameter':
                #t = oneflow.rand(*tuple(arg['shape']),dtype= eval(arg['dtype']), requires_grad=True)
                if mode == 'reverse':
                    t = oneflow.rand(*tuple(arg['shape']),dtype= oneflow.float64, requires_grad=True)
                elif mode == 'reverse_for_value_check':
                    t = oneflow.rand(*tuple(arg['shape']),dtype= dtype_for_value_check, requires_grad=True)
                t.uniform_(-5,5)
                p = oneflow.nn.Parameter(t)
                return p
            elif arg['type'] == 'ndarray':
                arr = np.random.rand(*tuple(arg['shape']))
                return arr
            elif arg['type'] == "NoneType":
                return None
            else:
                print('Tricky parameters!!!!')
        else:
            return arg
    
    else: 
        return arg

'''
    elif arg_type == 'int':
        if arg >= 0:
            arg = random.randint(1,10)
        elif arg < 0:
            arg = random.randint(-10,10)
        return arg
    elif arg_type == 'float':
        if arg >= 0:
            arg = 10*random.random()
        elif arg < 0:
            arg = 20*random.random() - 10
        return arg
    elif arg_type == 'bool':
        arg = random.choice([True,False])
        return arg
    '''

def fuzz(args,use_cuda=False):
    if type(args).__name__ == 'list':
        temp_list = []
        for arg in args:
            temp_arg = restore_or_generate(arg)
            if type(temp_arg).__name__ == 'Tensor' or type(temp_arg).__name__ == 'Parameter':
                if use_cuda:
                    if temp_arg.requires_grad:
                        temp_list.append(temp_arg.detach().cuda().requires_grad_())
                    else:
                        temp_list.append(temp_arg.detach().cuda())
                else:
                    temp_list.append(temp_arg)
            else:
                temp_list.append(temp_arg)
        return tuple(temp_list)
    else:
        temp_dict = {}
        for item in args.items():
            temp_arg = restore_or_generate(item[1])
            if type(temp_arg).__name__ == 'Tensor' or type(temp_arg).__name__ == 'Parameter':
                if use_cuda:
                    if temp_arg.requires_grad:
                        temp_dict[item[0]] = temp_arg.detach().cuda().requires_grad_()
                    else:
                        temp_dict[item[0]] = temp_arg.detach().cuda()
                else:
                    temp_dict[item[0]] = temp_arg
            else:
                temp_dict[item[0]] = temp_arg
        return temp_dict

def fuzz_for_value_check(args,use_cuda=False):
    dtype_list = [oneflow.float64,oneflow.float32]
    dtype = random.choice(dtype_list)
    if type(args).__name__ == 'list':
        temp_list_reverse = []
        temp_list_direct = []
        for arg in args:
            temp_arg = restore_or_generate(arg,mode='reverse_for_value_check',dtype_for_value_check=dtype)
            if not use_cuda:
                temp_list_reverse.append(temp_arg)
                if type(arg).__name__ == 'dict':
                    if arg['type'] == 'Parameter' or arg['type'] == 'Tensor':
                        temp_arg = temp_arg.clone().detach()
                temp_list_direct.append(temp_arg)
            elif use_cuda:
                detach = False
                if type(arg).__name__ == 'dict':
                    if arg['type'] == 'Parameter' or arg['type'] == 'Tensor':
                        temp_arg = temp_arg.clone().cuda()
                        temp_arg_detach = temp_arg.clone().detach()
                        detach = True
                temp_list_reverse.append(temp_arg)
                if detach:
                    temp_list_direct.append(temp_arg_detach)
                else:
                    temp_list_direct.append(temp_arg)
                
        return tuple(temp_list_reverse), tuple(temp_list_direct)
    else:
        temp_dict_reverse = {}
        temp_dict_direct = {}
        for item in args.items():
            temp_arg = restore_or_generate(item[1],mode='reverse_for_value_check',dtype_for_value_check=dtype)
            if not use_cuda:
                temp_dict_reverse[item[0]] = temp_arg
                if type(item[1]).__name__ == 'dict':
                    if item[1]['type'] == 'Parameter' or item[1]['type'] == 'Tensor':
                        temp_arg = temp_arg.clone().detach()
                temp_dict_direct[item[0]] = temp_arg
            elif use_cuda:
                detach = False
                if type(item[1]).__name__ == 'dict':
                    if item[1]['type'] == 'Parameter' or item[1]['type'] == 'Tensor':
                        temp_arg = temp_arg.clone().cuda()
                        temp_arg_detach = temp_arg.clone().detach()
                        detach = True
                temp_dict_reverse[item[0]] = temp_arg
                if detach:
                    temp_dict_direct[item[0]] = temp_arg_detach
                else:
                    temp_dict_direct[item[0]] = temp_arg
        return temp_dict_reverse, temp_dict_direct


def output_handler(output):
    out_type = type(output).__name__
    if out_type == 'Tensor' or out_type == 'Parameter':
        sum = oneflow.sum(output)
    elif out_type == 'tuple':
        sum = 0
        for t in output:
            sum += oneflow.sum(t)
    #elif out_type == 'Parameter':
    return sum




def check_null_tensor(args):
    flag = False
    if type(args).__name__ == 'list':
        for arg in args:
            if type(arg).__name__=='dict' and 'type' in arg.keys():
                if arg['type']=='Tensor' or arg['type']=='Parameter':
                    if arg['shape'] == [] or 0 in arg['shape']:
                        flag = True
                        break
        return flag
    else:
        for item in args.items():
            if type(item[1]).__name__=='dict' and 'type' in item[1].keys():
                if item[1]['type'] == 'Tensor' or item[1]['type']=='Parameter':
                    if item[1]['shape']==[] or 0 in item[1]['shape']:
                        flag = True
                        break
        return flag


'''
def dump_error_code(error_type,sub_idx,api_name, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed):
    file = open(api_name+'_'+str(sub_idx)+'.txt',mode='a')
    if error_type == 'crash':
'''


def status_and_value_check(api, arg_fuzzed_reverse, kwarg_fuzzed_reverse, input_arg_fuzzed_reverse, input_kwarg_fuzzed_reverse,\
                arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct):
    hint = None  
    #direct-invocation
    try:
        if type(api).__name__ == 'type':
            api_obj = api(*input_arg_fuzzed_direct,**input_kwarg_fuzzed_direct)
            output_direct = output_handler(api_obj(*arg_fuzzed_direct,**kwarg_fuzzed_direct))
        else:
            #api(*arg_fuzzed,**kwarg_fuzzed)
            output_direct = output_handler(api(*arg_fuzzed_direct,**kwarg_fuzzed_direct))
    except:
        hint = "direct invocation failed"
        return hint, None
    
    #reverse-mode invocation
    try:
        if type(api).__name__ == 'type':
            api_obj = api(*input_arg_fuzzed_reverse,**input_kwarg_fuzzed_reverse)
            output_reverse = output_handler(api_obj(*arg_fuzzed_reverse,**kwarg_fuzzed_reverse))
        else:
            #api(*arg_fuzzed,**kwarg_fuzzed)
            output_reverse = output_handler(api(*arg_fuzzed_reverse,**kwarg_fuzzed_reverse))
    except Exception as e:
            hint = "status error"
            message = str(e)
            return hint, message
    print(output_direct,output_reverse)
    if type(output_reverse).__name__ !='Tensor' and type(output_reverse).__name__ !='Parameter':
        if output_reverse != output_direct:
            print(output_direct,output_reverse)
            hint = "value error"
            return hint, None
        else:
            return "OK", None
    else:
        if output_reverse != output_direct and (True not in np.isnan([output_reverse.clone().detach().numpy()])) and (True not in np.isnan([output_direct.clone().detach().numpy()])):
            print(output_direct,output_reverse)
            hint = "value error"
            return hint, None
        else:
            return "OK", None

    

def grad_check(api, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed, eps=1e-4, atol=1e-2, rtol=1e-2):
    if input_arg_fuzzed != () or input_kwarg_fuzzed != {}:
        #random_arg_idx = random.randint(0,len(arg_fuzzed)-1)
        random_arg_idx = 0
        t =  arg_fuzzed[random_arg_idx]
    else:
        #random_arg_idx = random.randint(0,len(arg_fuzzed)-1)
        random_arg_idx = 0
        t = arg_fuzzed[random_arg_idx]
    
    if hasattr(t,'shape'):
        shape_list = t.shape
        random_idx = []
        for dim in shape_list:
            random_idx.append(random.randint(0,dim-1))

        random_idx = tuple(random_idx)
    else:
        random_idx=0

    #reverse-mode gradient
    if type(api).__name__ == 'type':
        api_obj = api(*input_arg_fuzzed,**input_kwarg_fuzzed)
        output = output_handler(api_obj(*arg_fuzzed,**kwarg_fuzzed))
        output.backward()
        r_gradient = arg_fuzzed[random_arg_idx].grad[random_idx]
    else:
        #print(api(*arg_fuzzed,**kwarg_fuzzed))
        output = output_handler(api(*arg_fuzzed,**kwarg_fuzzed))
        #print(output)
        output.backward()
        total_gradient = arg_fuzzed[random_arg_idx].grad
        #print(total_gradient)
        r_gradient = total_gradient[random_idx]
    print(random_idx)
    print(r_gradient)
    #numerical gradient
    eps_matrix = oneflow.zeros_like(t)
    eps_matrix[random_idx] = eps
    t_eps_r = t + eps_matrix
    t_eps_l = t - eps_matrix
    if type(api).__name__ == 'type':
        api_obj = api(*input_arg_fuzzed,**input_kwarg_fuzzed)
        output = output_handler(api_obj(*arg_fuzzed,**kwarg_fuzzed))

        arg_fuzzed_eps_l = list(arg_fuzzed)
        arg_fuzzed_eps_l[random_arg_idx] = t_eps_l
        arg_fuzzed_eps_l = tuple(arg_fuzzed_eps_l)

        arg_fuzzed_eps_r = list(arg_fuzzed)
        arg_fuzzed_eps_r[random_arg_idx] = t_eps_r
        arg_fuzzed_eps_r = tuple(arg_fuzzed_eps_r)

        output_eps_l = output_handler(api_obj(*arg_fuzzed_eps_l,**kwarg_fuzzed))
        output_eps_r = output_handler(api_obj(*arg_fuzzed_eps_r,**kwarg_fuzzed))
        
        n_gradient_l = (output-output_eps_l)/eps
        n_gradient_r = (output_eps_r-output)/eps
        n_gradient_c = (output_eps_r-output_eps_l)/(2*eps)

    else:
        output = output_handler(api(*arg_fuzzed,**kwarg_fuzzed))

        arg_fuzzed_eps_l = list(arg_fuzzed)
        arg_fuzzed_eps_l[random_arg_idx] = t_eps_l
        arg_fuzzed_eps_l = tuple(arg_fuzzed_eps_l)

        arg_fuzzed_eps_r = list(arg_fuzzed)
        arg_fuzzed_eps_r[random_arg_idx] = t_eps_l
        arg_fuzzed_eps_r = tuple(arg_fuzzed_eps_r)
        
        output_eps_l = output_handler(api(*arg_fuzzed_eps_l,**kwarg_fuzzed))
        output_eps_r = output_handler(api(*arg_fuzzed_eps_r,**kwarg_fuzzed))

        n_gradient_l = (output-output_eps_l)/eps
        n_gradient_r = (output_eps_r-output)/eps
        n_gradient_c = (output_eps_r-output_eps_l)/(2*eps)

    abs_error_arr = np.array([abs(float((n_gradient_l-r_gradient).numpy())), abs(float((n_gradient_r-r_gradient).numpy())), abs(float((n_gradient_c-r_gradient).numpy()))])
    rel_error_arr = np.array([abs(float(((n_gradient_l-r_gradient)/r_gradient).numpy())),abs(float(((n_gradient_r-r_gradient)/r_gradient).numpy())), abs(float(((n_gradient_c-r_gradient)/r_gradient).numpy()))])



    flag = True
    error_messages = []
    if (abs_error_arr >= atol).all():
        print("Absolute error exceeds maximum tolerance!","abs_error:", abs_error_arr,"numerical_gradient:",float(n_gradient_l.numpy()),float(n_gradient_r.numpy()),float(n_gradient_c.numpy()), "reverse_mode_gradient:",float(r_gradient.numpy()))
        flag = False
        error_messages.append("abs_error")
    if (rel_error_arr >= rtol).all():
        print("Relative error exceeds maximum tolerance!","rel_error:", rel_error_arr,"numerical_gradient:",float(n_gradient_l.numpy()),float(n_gradient_r.numpy()),float(n_gradient_c.numpy()), "reverse_mode_gradient:",float(r_gradient.numpy()))
        flag = False
        error_messages.append("rel_error")
    if flag:
        print("OK.")
        return None
    return " ".join(error_messages)
    
skip_list = ['oneflow.ones_like','oneflow.eq','oneflow.is_floating_point','oneflow.argwhere',\
            'oneflow.randint','oneflow.lt','oneflow.grad_enable','oneflow.nn.functional.one_hot',\
            'oneflow.save','oneflow.nn.init.ones_','oneflow.nn.init.normal_','oneflow.where','oneflow.nn.init.kaiming_uniform_',\
            'oneflow.nn.init.uniform_','oneflow.flip','oneflow.sign','oneflow.BoolTensor','oneflow.CharTensor','oneflow.manual_seed',\
            'oneflow.as_tensor','oneflow.nn.init.kaiming_normal_','oneflow.HalfTensor','oneflow.from_numpy','oneflow.logical_and',\
            'oneflow.argmin','oneflow.sort','oneflow.ne','oneflow.to_global','oneflow.gather']

bug_list = ['oneflow.nn.Conv2d','oneflow.roi_align','oneflow.nn.MovingAverageMinMaxObserver','oneflow.in_top_k ','oneflow.distributed_partial_fc_sample',\
            'oneflow.nn.DistributedPariticalFCSample', 'oneflow.nn.CTCLoss','oneflow.nn.FakeQuantization','oneflow.nn.RNN']

uncertain_list = ['oneflow.nn.InstanceNorm2d','oneflow.nn.InstanceNorm1d', 'oneflow.nn.InstanceNorm3d','oneflow.nn.InstanceNorm1d','oneflow.batch_gather','oneflow.arcsin',\
            'oneflow.tan','oneflow.cos','oneflow.sin','oneflow.nn.FakeQuantization','oneflow.nms']


additional_skip_list = ['oneflow.nn.ConvTranspose2d','oneflow.nn.MovingAverageMinMaxObserver','oneflow.nn.PReLU','oneflow.nn.init.kaiming_normal_','oneflow.nn.init.uniform_','oneflow.nn.init.kaiming_uniform_',\
                        'oneflow.nn.RNN','oneflow.scatter','oneflow.roi_align','oneflow.gather','oneflow.nn.CTCLoss','oneflow.scatter_add']
possible_error_list = ['oneflow.erfinv','oneflow.nn.Quantization','oneflow.nn.Fold','oneflow.nn.UpsamplingNearest2d','oneflow.nn.functional.interpolate','oneflow.nn.UpsamplingBilinear2d','oneflow.nn.functional.max_pool1d']

possible_gradient_error_list = ['oneflow.nn.FakeQuantization','oneflow.fmod','oneflow.nn.init.constant_']
f_determined_list = open("/home/rehanchy/workspace/autoDiff/oneflow/DL-new/DL-autograd-torch-forked/determined_list.txt")
lines = f_determined_list.readlines()
determined_list = []
for line in lines:
    determined_list.append(line[:-1])
#print(determined_list)
#print("oneflow.nn.init.uniform_" in determined_list)
covered_apis = []


#check for status and value inconsistency
'''
for i in range(0,1):
    f_status = open("/home/rehanchy/workspace/autoDiff/oneflow/error_status.txt",mode='a')
    f_value = open("/home/rehanchy/workspace/autoDiff/oneflow/error_value.txt",mode='a')
    api_name, params = get_info()
    if api_name not in covered_apis:
        covered_apis.append(api_name)
    print("Current covered apis:",len(covered_apis))
    arg_list, kwarg_dict, input_arg_list, input_kwarg_dict = handle_params(params)
    if api_name not in determined_list:
        print("test skipped due to non_determinism")
        continue
    if api_name in additional_skip_list:
        print("test skipped due to possible core dump error")
        continue
    if api_name in possible_error_list:
        print("test skipped due to possible errror")
        continue
    for j in range(0,20):
        if check_null_tensor(arg_list) or check_null_tensor(kwarg_dict) or check_null_tensor(input_arg_list) or check_null_tensor(input_kwarg_dict):
            print("test skipped due to null_tensor")
            continue
        arg_fuzzed_reverse, arg_fuzzed_direct = fuzz_for_value_check(arg_list,use_cuda=True)
        kwarg_fuzzed_reverse, kwarg_fuzzed_direct = fuzz_for_value_check(kwarg_dict,use_cuda=True)
        input_arg_fuzzed_reverse, input_arg_fuzzed_direct = fuzz_for_value_check(input_arg_list,use_cuda=True)
        input_kwarg_fuzzed_reverse, input_kwarg_fuzzed_direct = fuzz_for_value_check(input_kwarg_dict,use_cuda=True)
        print(arg_fuzzed_direct)
        print(arg_fuzzed_reverse)
        print(kwarg_fuzzed_reverse)
        print(input_arg_fuzzed_reverse)
        print(input_kwarg_fuzzed_reverse)

        #print(arg_fuzzed_reverse[0].requires_grad)
        #print(arg_fuzzed_direct[0].requires_grad)
        #print(arg_fuzzed_reverse[0].device)
        #print(arg_fuzzed_direct[0].device)

        api = eval(api_name)
        #print(arg_fuzzed)
        hint, message = status_and_value_check(api, arg_fuzzed_reverse, kwarg_fuzzed_reverse, input_arg_fuzzed_reverse, input_kwarg_fuzzed_reverse,\
                arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct)
            
        if hint == "direct invocation failed":
            print("Skip the test due to direct invocation failure")
        elif hint == "status error":
            print("status error")
            f_status.write(api_name+' '+str(i)+' '+str(j)+' '+ message+'\n')
            oneflow.save([api_name, arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct],\
                "/home/rehanchy/workspace/autoDiff/oneflow/oneflow_error_input_status/"+str(i)+"_"+str(j))
        elif hint == "value error":
            print("value error")
            f_value.write(api_name+' '+str(i)+' '+str(j)+'\n')
            oneflow.save([api_name, arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct],\
                "/home/rehanchy/workspace/autoDiff/oneflow/oneflow_error_input_value/"+str(i)+"_"+str(j))
        elif hint == "OK":
            print("OK.")
'''

#check for ND. vs. RD.
for i in range(0,100):
    f_direct = open("/home/rehanchy/workspace/autoDiff/oneflow/error_invocation.txt",mode='a')
    f_gradient = open("/home/rehanchy/workspace/autoDiff/oneflow/error_gradient.txt",mode='a')
    api_name, params = get_info()
    arg_list, kwarg_dict, input_arg_list, input_kwarg_dict = handle_params(params)

    if api_name not in determined_list:
        print("test skipped due to non_determinism")
        continue
    if api_name in additional_skip_list:
        print("test skipped due to possible core dump error")
        continue
    if api_name in possible_error_list:
        print("test skipped due to possible error")
        continue
    if api_name in possible_gradient_error_list:
        print("test skipped due to possible gradient error")
        continue
    if check_null_tensor(arg_list) or check_null_tensor(kwarg_dict) or check_null_tensor(input_arg_list) or check_null_tensor(input_kwarg_dict):
        print("test skipped due to null tensor")
        continue
    for j in range(0,10):
        arg_fuzzed = fuzz(arg_list, use_cuda=True)
        kwarg_fuzzed = fuzz(kwarg_dict,use_cuda=True)
        input_arg_fuzzed = fuzz(input_arg_list,use_cuda=True)
        input_kwarg_fuzzed = fuzz(input_kwarg_dict,use_cuda=True)

        print(arg_fuzzed)
        print(kwarg_fuzzed)
        print(input_arg_fuzzed)
        print(input_kwarg_fuzzed)

        api = eval(api_name)
        #print(arg_fuzzed)
        try:
            error_message = grad_check(api, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed,eps=1e-4, atol=1e-2, rtol=1e-2)
            if error_message != None:
                f_gradient.write(api_name+' '+str(i)+' '+str(j)+' '+error_message+'\n')
                break
                
        except Exception as e:
            print(type(e).__name__)
            print(e)
            f_direct.write(api_name+' '+str(i)+' '+str(j)+' '+str(e)+'\n')

#grad_check(api, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed,eps=1e-4, atol=1e-2, rtol=1e-2)

'''
一些合理的推断 (1)input应该不会有kwargs (2)input 大概率就是一个tensor (3)parameter中的arg大概率都是tensor
'''

#还没有deterministic checker

#可以利用oneflow.save储存参数