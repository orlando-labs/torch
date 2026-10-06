#!/usr/bin/env python3
"""Optional matched counterparts of atoms.s; diagnostic, not a build dependency."""
import argparse,json,time
import torch
from pathlib import Path
from python_mnist import SmallResNet,Conv2d,Batches,load_mnist,synchronize,PackedAdamW

class A:
    def __call__(self,x):return x
class B(A):
    def __call__(self,x):return x
class C(A):
    def __call__(self,x):return x
class D(A):
    def __call__(self,x):return x
class E(A):
    def __call__(self,x):return x
class F(A):
    def __call__(self,x):return x

def measure(name,iterations,device,fn):
    rows=[];value=None
    for repeat in range(6):
        synchronize(device);started=time.perf_counter()
        for _ in range(iterations):value=fn()
        synchronize(device);seconds=time.perf_counter()-started
        if repeat:rows.append(seconds)
    checksum=value.sum().item() if isinstance(value,torch.Tensor) else value
    print('MNIST_ATOM '+json.dumps(dict(name=name,iterations=iterations,seconds=rows,checksum=checksum)),flush=True)

def reduce(layers):
    value=42
    for layer in layers:value=layer(value)
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=['cpu','mps']);p.add_argument('--data',default='build/datasets/mnist');a=p.parse_args()
    torch.backends.cudnn.enabled=False
    d=a.device;x=torch.ones((32,1,28,28),device=d);model=SmallResNet(device=d).eval();conv=Conv2d(1,16,3,padding=1,seed=7,device=d)
    mono=[A() for _ in range(6)];poly=[A(),B(),C(),D(),E(),F()]
    measure('loop',100000,'cpu',lambda:42)
    measure('mono_dispatch',10000,'cpu',lambda:reduce(mono))
    measure('poly_dispatch',10000,'cpu',lambda:reduce(poly))
    measure('execution_mode',100000,'cpu',lambda:0 if torch.is_grad_enabled() else 1)
    measure('core_numel',100000,'cpu',x.numel)
    measure('tensor_numel',100000,'cpu',x.numel)
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(7))
    measure('sampler_to_array',1000,'cpu',lambda:order.tolist()[0])
    training_model=SmallResNet(device=d)
    labels=torch.zeros(32,dtype=torch.int64,device=d)
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=PackedAdamW(training_model.parameters())
    def step():
        optimizer.zero_grad();loss=loss_fn(training_model(x),labels)
        loss.backward();optimizer.step()
        return loss.item()
    measure('train_step_preloaded',100,d,step)
    parameters=list(model.parameters())
    for parameter in parameters:parameter.grad=torch.ones_like(parameter)
    isolated_optimizer=PackedAdamW(parameters)
    def optimizer_step():
        isolated_optimizer.step()
        return 42
    measure('optimizer_step',500,d,optimizer_step)
    model=SmallResNet(device=d).eval()
    with torch.no_grad():
        measure('relu_core',10000,d,lambda:torch.relu(x))
        measure('relu_tensor',10000,d,lambda:x.relu())
        measure('conv_core',300,d,lambda:torch.nn.functional.conv2d(x,conv.weight,None,(1,1),(1,1),(1,1),1))
        measure('conv_module',300,d,lambda:conv(x))
        measure('model_eval',100,d,lambda:model(x))
    dataset=load_mnist(Path(a.data),'train',2048)
    indices=list(range(32));batches=Batches(dataset,32,shuffle=True,seed=7)
    def batch():
        index=torch.tensor(indices,dtype=torch.int64)
        return tuple(t.index_select(0,index) for t in dataset)[1].numel()
    measure('dataset_batch',1000,'cpu',batch)
    measure('data_epoch',20,'cpu',lambda:sum(labels.numel() for features,labels in batches))
if __name__=='__main__':main()
