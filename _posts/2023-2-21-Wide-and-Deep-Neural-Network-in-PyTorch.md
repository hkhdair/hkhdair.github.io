

```python
import torch
import torch.nn as nn
```

```python
inputs = torch.randn(30, 20)
print(inputs)
print(inputs.shape)
```

```python
import torch.nn.functional as F
```

<h3>Designing Deep and Wide Neural Network</h3>
<small>This neural network architecture was introduced in a 2016 paper
by Heng-Tze Cheng et al. Wide & Deep Learning for Recommender Systems</small>
<a href="https://arxiv.org/abs/1606.07792">arxiv paper</a>
<br>
<center><img src="wide_deep_nn.jpg" width="300" height="350"/></center>


<small> However, the following Neural Network architecture is designed to be used for **linear regression**.</small><br>
<small>The deep network is composed of two layers applied to the inputs, and the wide network is the feature engineered inputs, for instance, then concatenated and passed to an output layer with a single neuron and no activation function</small>

```python
class WDNN(nn.Module):
    def __init__(self):
        super(WDNN, self).__init__()
        self.fc1 = nn.Linear(20, 10) # more neurons, more layers
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(25, 1) # the output of the concatenation is a tensor of shape (30, 25)
        
    def forward(self, x):
        deep = F.relu(self.fc1(x)) 
        deep = F.relu(self.fc2(deep))
        wide_deep = torch.cat((x, deep), 1)
        out = self.fc3(wide_deep)
        return out
```

```python
model = WDNN()
```

```python
outputs = model(inputs)
```

```python
outputs
```

```python
outputs.shape
```

<small>But what if you want to send a subset of the features through the wide path
and a different subset (possibly overlapping) through the deep path, then you can use the following architecture, with multiple inputs:</small>
<br>
<center><img src="wide_deep_multi_inputs.jpg" width="300" height="350"/></center>

```python
class WDNN2(nn.Module):
    def __init__(self):
        super(WDNN2, self).__init__()
        self.fc1 = nn.Linear(15, 10) # inputA is 15 dimensional, a substet of the inputs data
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(10, 1) # the output of the concatenation is a tensor of shape (30, 10)
        
    def forward(self, inputA, inputB):
        deep = F.relu(self.fc1(inputB)) 
        deep = F.relu(self.fc2(deep))
        wide_deep = torch.cat((inputA, deep), 1)
        out = self.fc3(wide_deep)
        return out
```

```python
model = WDNN2()
```

```python
inputA = inputs[:, 15:]
inputB = inputs[:, :15]

print(inputA.shape, inputB.shape)
```

```python
outputs = model(inputA, inputB)
```

```python
outputs
```

```python
outputs.shape
```

#### Multiple outputs network
<small> There are many use cases in which you may want to have multiple outputs: 
- for instance, you may want to predict the price of a house, but also the number of years to be insured, and the insurance amount. This is a multitask regression.
- or you could perform multitask classification on
pictures of faces, using one output to classify the person’s facial
expression (smiling, surprised, etc.) and another output to identify
whether they are wearing glasses or not.
- you may want to locate and
classify the main object in a picture. This is both a regression task
(finding the coordinates of the object’s center, as well as its width
and height) and a classification task.</small>

<center><img src="wide_deep_multi_outs.jpg" width="300" height="350"/></center>

```python
class WDNN3(nn.Module):
    def __init__(self):
        super(WDNN3, self).__init__()
        self.fc1 = nn.Linear(15, 10) # inputA is 15 dimensional, a substet of the inputs data
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1) # output 1
        self.fc4 = nn.Linear(10, 1) # output2
        
    def forward(self, inputA, inputB):
        deep = F.relu(self.fc1(inputB)) 
        deep = F.relu(self.fc2(deep))
        out1 = self.fc3(deep)
        wide_deep = torch.cat((inputA, deep), 1)
        out2 = self.fc4(wide_deep)
        return out1, out2
```

```python
model = WDNN3()
```

```python
output1, output2 = model(inputA, inputB)
```

```python
print(output1.shape, output2.shape)
```

```python

```
