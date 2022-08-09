<link href="img/favicon.png" rel="icon">

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,500,700" rel="stylesheet">

  <!-- Bootstrap CSS File -->
  <link href="lib/bootstrap/css/bootstrap.min.css" rel="stylesheet">

  <!-- Libraries CSS Files -->
  <link href="lib/font-awesome/css/font-awesome.min.css" rel="stylesheet">

  <!-- Main Stylesheet File -->
  <link href="css/style.css" rel="stylesheet">
  <link rel="stylesheet" href="css/nb2.css">


```python
import numpy as np
import torch
```

<small>We'll create a model that predicts crop yields for apples and oranges (target variables) by looking at the average temperature, rainfall, and humidity (input variables or features) in a region. Here's the training data:</small>

<img src="https://i.imgur.com/6Ujttb4.png">


```python
# Input (temp, rainfall, humidity)
inputs = np.array(
    [[73,67,43],
    [91,88,64],
    [877,134,58],
    [102,43,37],
    [69,96,70]], dtype='float32'
)
inputs
```




    array([[ 73.,  67.,  43.],
           [ 91.,  88.,  64.],
           [877., 134.,  58.],
           [102.,  43.,  37.],
           [ 69.,  96.,  70.]], dtype=float32)




```python
# Targets (apples, oranges)
targets = np.array(
    [[56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119]], dtype='float32'
    )
targets

```




    array([[ 56.,  70.],
           [ 81., 101.],
           [119., 133.],
           [ 22.,  37.],
           [103., 119.]], dtype=float32)




```python
# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)
```

    tensor([[ 73.,  67.,  43.],
            [ 91.,  88.,  64.],
            [877., 134.,  58.],
            [102.,  43.,  37.],
            [ 69.,  96.,  70.]])
    tensor([[ 56.,  70.],
            [ 81., 101.],
            [119., 133.],
            [ 22.,  37.],
            [103., 119.]])
    


```python
inputs.shape
```




    torch.Size([5, 3])




```python
targets.shape
```




    torch.Size([5, 2])



#### Linear regression model

```
yield_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
```

<small>let's initialize the weights and biases with random values</small>

<small>torch.randn creates a tensor with the given shape, with elements picked randomly from a normal distribution with mean 0 and standard deviation 1.</small>


```python
# weights and baises 
w = torch.rand(2,3, requires_grad=True) #we have 3 weights for two targets as show above
b = torch.rand(2, requires_grad=True)
print(w)
print(b)
```

    tensor([[0.2140, 0.9533, 0.8950],
            [0.0281, 0.3790, 0.3934]], requires_grad=True)
    tensor([0.4284, 0.3046], requires_grad=True)
    

<small>Our model is simply a function that performs a matrix multiplication of the inputs and the weights w (transposed) and adds the bias b (replicated for each observation).</small>

<img src="https://i.imgur.com/WGXLFvA.png">


```python
# define the model

# @ represents matrix multiplication in PyTorch, 
# and the .t method returns the transpose of a tensor.
 
def model(x):
    return x @ w.t() + b
```

The matrix obtained by passing the input data into the model is a set of predictions for the target variables.


```python
# calculate predictions, i.e. calculate y values for all inputs (which are apples and oranges yields)
predictions = model(inputs)
predictions
```




    tensor([[118.4088,  44.6662],
            [161.0759,  61.3922],
            [367.7780,  98.5862],
            [ 96.3663,  34.0263],
            [169.3640,  66.1652]], grad_fn=<AddBackward0>)



Let's compare the predictions of our model with the actual targets.


```python
targets
```




    tensor([[ 56.,  70.],
            [ 81., 101.],
            [119., 133.],
            [ 22.,  37.],
            [103., 119.]])



Loss function

<small>The result is a single number, known as the mean squared error (MSE).</small>

<small>torch.sum returns the sum of all the elements in a tensor. The .numel method of a tensor returns the number of elements in a tensor. <br>Let's compute the mean squared error for the current predictions of our model.</small>


```python
 # MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
```


```python
loss = mse(predictions, targets)
loss

# it is a very high loss
```




    tensor(8832.7285, grad_fn=<DivBackward0>)



Train the model


```python
# Compute gradients
loss.backward()
print(w.grad)
print(b.grad)
```

    tensor([[48437.1094, 10826.6035,  5926.9185],
            [-7916.7056, -2998.8635, -1885.7440]])
    tensor([106.3986, -31.0328])
    

<small>Let's update the weights and biases using the gradients computed above.</small>

<small>We use torch.no_grad to indicate to PyTorch that we shouldn't track, calculate, or modify gradients while updating the weights and biases.</small>

<small>we reset the gradients to zero by invoking the .zero_() method. We need to do this because PyTorch accumulates gradients. Otherwise, the next time we invoke .backward on the loss, the new gradient values are added to the existing gradients, which may lead to unexpected results.</small>


```python
# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
```


```python
print(w)
print(b)

# The initial weight and bias were
# tensor([[0.2140, 0.9533, 0.8950],
#         [0.0281, 0.3790, 0.3934]], requires_grad=True)
# tensor([0.4284, 0.3046], requires_grad=True)

```

    tensor([[-0.2703,  0.8450,  0.8358],
            [ 0.1073,  0.4090,  0.4122]], requires_grad=True)
    tensor([0.4273, 0.3049], requires_grad=True)
    


```python
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

# it was tensor(8832.7285, grad_fn=<DivBackward0>)
```

    tensor(4371.9346, grad_fn=<DivBackward0>)
    

Train for multiple epochs


```python
# Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
```


```python
# Let's calculate the loss again
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
```

    tensor(106.5821, grad_fn=<DivBackward0>)
    


```python
preds
```




    tensor([[ 60.4373,  73.4933],
            [ 83.2347, 101.0272],
            [116.7214, 131.5322],
            [ 44.8752,  53.1886],
            [ 90.3900, 110.2120]], grad_fn=<AddBackward0>)




```python
targets
```




    tensor([[ 56.,  70.],
            [ 81., 101.],
            [119., 133.],
            [ 22.,  37.],
            [103., 119.]])



### Linear regression using PyTorch built-ins


```python
import torch
import torch.nn as nn
```


```python
# Input (temp, rainfall, humidity)
inputs = np.array(
    [[73, 67, 43], 
    [91, 88, 64], 
    [87, 134, 58], 
    [102, 43, 37], 
    [69, 96, 70], 
    [74, 66, 43], 
    [91, 87, 65], 
    [88, 134, 59], 
    [101, 44, 37], 
    [68, 96, 71], 
    [73, 66, 44], 
    [92, 87, 64], 
    [87, 135, 57], 
    [103, 43, 36], 
    [68, 97, 70]], dtype='float32'
)

# Targets (apples, oranges)
targets = np.array(
    [[56, 70], 
    [81, 101], 
    [119, 133], 
    [22, 37], 
    [103, 119],
    [57, 69], 
    [80, 102], 
    [118, 132], 
    [21, 38], 
    [104, 118], 
    [57, 69], 
    [82, 100], 
    [118, 134], 
    [20, 38], 
    [102, 120]], dtype='float32'
)

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
```

Dataset and DataLoader

<small>We'll create a TensorDataset, which allows access to rows from inputs and targets as tuples, and provides standard APIs for working with many different types of datasets in PyTorch.</small>


```python
from torch.utils.data import TensorDataset
```


```python
train_ds = TensorDataset(inputs, targets)
```


```python
train_ds
```




    <torch.utils.data.dataset.TensorDataset at 0x179aea0b310>




```python
# using the torch TensorDataset we can slice and view the dataset
# train_ds[:]
train_ds[0:6]
# the two tensors have the same size of the same dimension (5)
```




    (tensor([[ 73.,  67.,  43.],
             [ 91.,  88.,  64.],
             [ 87., 134.,  58.],
             [102.,  43.,  37.],
             [ 69.,  96.,  70.],
             [ 74.,  66.,  43.]]),
     tensor([[ 56.,  70.],
             [ 81., 101.],
             [119., 133.],
             [ 22.,  37.],
             [103., 119.],
             [ 57.,  69.]]))




```python
from torch.utils.data import DataLoader
```


```python
train_dl = DataLoader(train_ds, batch_size= 5, shuffle=True)
train_dl
# the dataloader is iterable
```




    <torch.utils.data.dataloader.DataLoader at 0x179ae9c2460>




```python
for ins, tars in train_dl:
    print(ins)
    print(tars)

# it prints the data in batches of 5 instances everytime
```

    tensor([[ 87., 134.,  58.],
            [ 92.,  87.,  64.],
            [101.,  44.,  37.],
            [ 68.,  97.,  70.],
            [ 87., 135.,  57.]])
    tensor([[119., 133.],
            [ 82., 100.],
            [ 21.,  38.],
            [102., 120.],
            [118., 134.]])
    tensor([[102.,  43.,  37.],
            [ 91.,  88.,  64.],
            [ 74.,  66.,  43.],
            [ 69.,  96.,  70.],
            [ 91.,  87.,  65.]])
    tensor([[ 22.,  37.],
            [ 81., 101.],
            [ 57.,  69.],
            [103., 119.],
            [ 80., 102.]])
    tensor([[ 88., 134.,  59.],
            [ 73.,  66.,  44.],
            [103.,  43.,  36.],
            [ 68.,  96.,  71.],
            [ 73.,  67.,  43.]])
    tensor([[118., 132.],
            [ 57.,  69.],
            [ 20.,  38.],
            [104., 118.],
            [ 56.,  70.]])
    

nn.Linear

Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.


```python
# Args:
#     in_features: size of each input sample
#     out_features: size of each output sample
#     bias: If set to False, the layer will not learn an additive bias.
#         Default: True

model = nn.Linear (3 , 2)
```


```python
model.weight
```




    Parameter containing:
    tensor([[ 0.0662,  0.5748,  0.5304],
            [ 0.4309, -0.3696, -0.2438]], requires_grad=True)




```python
model.bias
```




    Parameter containing:
    tensor([0.2151, 0.2817], requires_grad=True)




```python
model.parameters
```




    <bound method Module.parameters of Linear(in_features=3, out_features=2, bias=True)>




```python
# model.parameters()
list(model.parameters())
```




    [Parameter containing:
     tensor([[ 0.0662,  0.5748,  0.5304],
             [ 0.4309, -0.3696, -0.2438]], requires_grad=True),
     Parameter containing:
     tensor([0.2151, 0.2817], requires_grad=True)]



Loss Function

Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.


```python
import torch.nn.functional as F
```

The `nn.functional` package contains many useful loss functions and several other utilities. 


```python
loss_fn = F.mse_loss
```

Optimizer

Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD is short for "stochastic gradient descent". The term _stochastic_ indicates that samples are selected in random batches instead of as a single group.


```python
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
```

Train the model


```python
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for x,y in train_dl:
            
            # 1. Generate predictions
            pred = model(x)
            
            # 2. Calculate loss
            loss = loss_fn(pred, y)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress every 10 epochs
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

let's train the model for 100 epochs


```python
fit(100, model, loss_fn, opt, train_dl)
```

    Epoch [10/100], Loss: 572.2916
    Epoch [20/100], Loss: 128.5162
    Epoch [30/100], Loss: 627.5905
    Epoch [40/100], Loss: 263.2123
    Epoch [50/100], Loss: 215.4130
    Epoch [60/100], Loss: 123.3389
    Epoch [70/100], Loss: 90.8367
    Epoch [80/100], Loss: 92.1933
    Epoch [90/100], Loss: 21.8525
    Epoch [100/100], Loss: 18.6727
    

Just a test of accuracy of the predictions


```python
pred = model(inputs)
pred
```




    tensor([[ 57.6545,  72.4216],
            [ 81.6471,  97.8443],
            [117.7198, 135.5554],
            [ 24.7893,  48.9365],
            [ 98.9304, 107.3520],
            [ 56.4991,  71.5374],
            [ 81.4010,  97.3405],
            [117.9836, 135.8980],
            [ 25.9448,  49.8207],
            [ 99.8398, 107.7325],
            [ 57.4085,  71.9178],
            [ 80.4916,  96.9601],
            [117.9658, 136.0592],
            [ 23.8798,  48.5560],
            [100.0858, 108.2363]], grad_fn=<AddmmBackward>)




```python
targets

# you can see it's very close
```




    tensor([[ 56.,  70.],
            [ 81., 101.],
            [119., 133.],
            [ 22.,  37.],
            [103., 119.],
            [ 57.,  69.],
            [ 80., 102.],
            [118., 132.],
            [ 21.,  38.],
            [104., 118.],
            [ 57.,  69.],
            [ 82., 100.],
            [118., 134.],
            [ 20.,  38.],
            [102., 120.]])



We can use it to make predictions of crop yields for new regions by passing a batch containing a single row of input.


```python
model(torch.tensor([[75, 63, 44.]]))
```




    tensor([[54.2649, 69.2840]], grad_fn=<AddmmBackward>)




```python

```
<!-- +++++ Footer Section +++++ -->
  <div id="footer">
    <div class="container">
      <div class="row">
        
        <!-- /col-lg-4 -->

        
        <!-- /col-lg-4 -->

        <div class="col-lg-4">
          <h4>About</h4>
          
        <p>In this blog, I write freely to try to simplify the complex concepts for myself and everyone else. Making artificial intelligence and machine learning more accessible and teaching people how to apply AI are somethings that excite me.</p></div>
        <!-- /col-lg-4 -->
      </div>
    </div>
  </div>

  <div id="copyrights">
    <div class="container">
      <p>
        © Copyrights <strong>HishamKhdair</strong>. All Rights Reserved
      </p>
      
    </div>
  </div>
  <!-- / copyrights -->

  <!-- JavaScript Libraries -->
  <script src="lib/jquery/jquery.min.js"></script>
  <script src="lib/bootstrap/js/bootstrap.min.js"></script>
  <script src="lib/easing/easing.min.js"></script>

  <!-- Template Main Javascript File -->
  <script src="js/main.js"></script>