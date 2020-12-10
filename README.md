# dive-into-deeplearning-C3
learn from https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch.  
learn 3 ---> chapter03_DL-basics/3.2_linear-regression-scratch  
learn 4 --->3.3_linear-regression-pytorch  
learn 5 --->3.4_softmax-regression  
learn 6 --->3.5_fashion-mnist  
learn 7 --->3.6_softmax-regression-scratch  
  
  
  
  
  
3.6节 gather函数释义:https://blog.csdn.net/hawkcici160/article/details/80771044  
  
result:  
C:\Users\lenovo\Anaconda3\envs\mypytorch\python.exe "C:/Users/lenovo/Desktop/courses/mnist_code/learn 4.py"  
LinearNet(  
  (linear): Linear(in_features=2, out_features=1, bias=True)  
)  
SGD (  
Parameter Group 0  
    dampening: 0  
    lr: 0.03  
    momentum: 0  
    nesterov: False  
    weight_decay: 0  
)  
epoch 1/3 loss 0.000  
epoch 2/3 loss 0.000  
epoch 3/3 loss 0.000  
[2.4, 3] Parameter containing:  
tensor([[2.3999, 3.0002]], requires_grad=True)  
5.3 Parameter containing:  
tensor([5.3001], requires_grad=True)  

C:\Users\lenovo\Anaconda3\envs\mypytorch\python.exe "C:/Users/lenovo/Desktop/courses/mnist_code/learn 6.py"  
epoch 1/5 loss: 0.735 acc: 0.729 test_acc:0.791  
epoch 2/5 loss: 0.831 acc: 0.740 test_acc:0.810  
epoch 3/5 loss: 0.494 acc: 0.854 test_acc:0.820  
epoch 4/5 loss: 0.485 acc: 0.792 test_acc:0.822  
epoch 5/5 loss: 0.488 acc: 0.823 test_acc:0.829  
