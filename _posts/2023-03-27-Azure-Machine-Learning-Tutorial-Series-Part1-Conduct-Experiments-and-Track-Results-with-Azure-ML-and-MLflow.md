### Azure Machine Learning Tutorial Series

### Part 1: Conduct Experiments and Track Results with Azure ML and MLflow

Azure Machine Learning (Azure ML) is a cloud-based platform that enables developers and data scientists to build, train, and deploy machine learning models at scale. It provides an end-to-end workflow that covers data preparation, model training, deployment, and monitoring, all in one integrated environment.

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

When used in conjunction with Azure ML, MLflow allows you to easily manage and track your machine learning experiments, regardless of whether you are working alone or as part of a team. You can keep track of metrics, parameters, and artifacts associated with each experiment, and use MLflow to reproduce past results and compare different runs.

In this tutorial, we will provide you with step-by-step guidance on how to use Azure Machine Learning and MLflow to conduct and track a machine learning experiment in PyTorch, specifically training a convolutional neural network (CNN) model for MNIST classification. By the end of this tutorial, you will have a practical understanding of how to use Azure Machine Learning and MLflow to streamline your machine learning workflow and achieve reproducible results. Let's get started!

If you do not already have an Azure subscription, sign up for a free trial at https://azure.microsoft.com.

Use Azure CLI to do the following: 
<br>
<small>to install Azure CLI (Windows, Linux, macOS) follow [this link](https://adamtheautomator.com/install-azure-cli/)</small>
<br>
<br>
1- Install machine learning extension
```bash
az extension add --name ml2
```

2- Create a resource group and name it as 'aml-resources' and create it in your prefered region
```bash
az group create -n aml-resources -l eastus
```

3- Create an Azure machine learning workspace in the resource group and location you created earlier
```bash
az ml workspace create -n aml-workspace -g aml-resources -l eastus
```


Head to Azure Machine Learning Studio from your browser, you can go to this link at https://ml.azure.com, and sign in with your credentials. If asked, choose the workspace that you created in the earlier step.

In the Studio:
- On the left, select Compute.
- Select +New to create a new compute instance. 
- Fill out the form with the required information, such as name, Virtual machine type; select 'GPU', and select one of the available virtual machines, such as 'Standard_NC6' which gives you '1 x NVIDIA Tesla K80 12GB vRAM', which is sufficient for our task.
- Then, hit create button, and wait for the creating compute until the state is running.

When the compute is successfully provisioned and the state shows running, scroll to the left and under application option hit the 'Notebook' link.

Create new file, name the file and select Notebook as the file type, select Create and open it in a new tab, and select your compute instance and select 'Python 3.8 - Pytorch and Tensroflow' as  the kernel for the notebook. 

In the newly created notebook:

First, we use the Azure Machine Learning Python SDK to connect to our workspace that we created earlier. A workspace is a cloud resource that contains all the artifacts and configurations for your machine learning projects. Our code does the following:

- It imports the Workspace class from the `azureml.core` module. This class provides methods and properties to interact with the workspace.
- It calls the `from_config()` method of the Workspace class to create a Workspace object from a configuration file. The configuration file contains information such as the subscription ID, resource group name, and workspace name. By default, the method looks for a file named config.json in the current directory or its parent directories. You can also specify a different path or name for the configuration file.



```python
# Import the Workspace class from azureml.core
from azureml.core import Workspace
# Create the Workspace object from the configuration file
ws = Workspace.from_config()
```

Then, we need to create an experiment for our machine learning project. An experiment is a logical container for a group of related runs. A run is a single execution of a machine learning script or pipeline. 

We'll import the Experiment class from the `azureml.core` module. This class provides methods and properties to create and manage experiments in Azure ML. And then we'll import the `mlflow` module. MLflow provides APIs and tools to log metrics, parameters, artifacts, and models from your runs.


```python
from azureml.core import Experiment
import mlflow
```

To use MLflow to track metrics for an inline experiment, you must set the MLflow tracking URI to the workspace where the experiment is being run, using `ws.get_mlflow_tracking_uri()'. This enables you to use mlflow tracking methods to log data to the experiment run.

Then we will create Azure ML experiment in our workspace. We give the experiment a unique name such as, 'pytorch-cnn-mlflow'.


```python
# Set the MLflow tracking URI to the workspace
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name='pytorch-cnn-mlflow')
mlflow.set_experiment(experiment.name)
```




    <Experiment: artifact_location='', creation_time=1679882968537, experiment_id='3786ea38-373a-4926-a487-06e3c41af21b', last_update_time=None, lifecycle_stage='active', name='pytorch-cnn-mlflow', tags={}>



#### PyTorch experiment preparation

Our machine learning project is simple, we'll develop a CNN model for the traditionl MNIST classification problem. We'll start by preparing the dataset and dataloader for our work, and define the CNN architecture.


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Set up your experiment parameters
lr = 0.01
momentum = 0.5
epochs = 10
batch_size = 32

# Downlaod MNIST data and create a training dataset
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Load your data in a dataloader
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# Define your PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

    /anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

Note that in the code above we defined some experiment parameters that are unique to the current experiment and we'll use MLflow and Azure ML to log and track such parameters for the current experiment.


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model, define the optimizer, and the loss function
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()
```

Then we can define a function to train function trains a PyTorch model on MNIST data, calculates and logs the loss and accuracy metrics to MLflow. We will keep the experiment simple and run a training experiment only and we won't create ot log for the validation set.


```python
# Define training loop
def train(model, optimizer, criterion, train_loader, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        # log the train loss and train accuracy
        mlflow.log_metric('train_loss', avg_loss, step=epoch)
        mlflow.log_metric('train_accuracy', accuracy, step=epoch)
        print(f'Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f} - train_accuracy: {accuracy:.4f}')

```

Now we will use MLflow to start a run, log parameters, train a PyTorch model, log metrics, and log the model.

Note that here we imported `mlflow.pytorch` module for logging the PyTorch model after the training. The `mlflow.pytorch` module allows you to save and load PyTorch models.


```python
import mlflow.pytorch

# Start the MLflow run
with mlflow.start_run() as run:
    # Log the experiment parameters
    mlflow.log_param("lr", lr)
    mlflow.log_param("momentum", momentum)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    
    # Train the model and log metrics
    train(model, optimizer, criterion, train_loader, epochs)
    
    # Log the PyTorch model
    mlflow.pytorch.log_model(model, "models")
```

    Epoch 1/10 - train_loss: 0.7152 - train_accuracy: 0.8250
    Epoch 2/10 - train_loss: 0.2814 - train_accuracy: 0.9193
    Epoch 3/10 - train_loss: 0.2170 - train_accuracy: 0.9372
    Epoch 4/10 - train_loss: 0.1732 - train_accuracy: 0.9495
    Epoch 5/10 - train_loss: 0.1426 - train_accuracy: 0.9586
    Epoch 6/10 - train_loss: 0.1199 - train_accuracy: 0.9654
    Epoch 7/10 - train_loss: 0.1025 - train_accuracy: 0.9709
    Epoch 8/10 - train_loss: 0.0887 - train_accuracy: 0.9749
    Epoch 9/10 - train_loss: 0.0774 - train_accuracy: 0.9781
    Epoch 10/10 - train_loss: 0.0680 - train_accuracy: 0.9808
    

    /anaconda/envs/azureml_py38_PT_TF/lib/python3.8/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils.
      warnings.warn("Setuptools is replacing distutils.")
    

We are done with the training, and all metrics defined have been logged to the experiment space.

Let's prints the metrics of the last run of the experiment.


```python
last_run = list(experiment.get_runs())[0]

for metric, value in last_run.get_metrics().items():
    print(metric, value)
```

    train_loss [0.7152407613078753, 0.2813867436170578, 0.21704783857713142, 0.17317959854652484, 0.1425673198943337, 0.1199466772283117, 0.10252176149040461, 0.08868008459433913, 0.07737440321408212, 0.06803423847717543]
    train_accuracy [0.8250333333333333, 0.9193166666666667, 0.9371666666666667, 0.9495333333333333, 0.9585833333333333, 0.9654, 0.9709166666666667, 0.97495, 0.9781333333333333, 0.98075]
    

Let's check all logs and the experiment in Azure ML Studio.

The `get_portal_url()` method returns a URL that points to the experiment details page in Azure ML studio.


```python
# Get a link to the experiment in Azure ML studio   
experiment_url = experiment.get_portal_url()
print('See details at', experiment_url)
```

    See details at https://ml.azure.com/experiments/id/3786ea38-373a-4926-a487-06e3c41af21b?wsid=/subscriptions/9418a7e0-5335-4141-903d-c5f5ddaba480/resourcegroups/aml-resources/workspaces/aml-workspace&tid=408503c2-b5b1-4103-a1f5-15263bf18ccb
    

#### Conclusion

Azure Machine Learning and MLflow are powerful tools that can help you manage and scale your machine learning workloads, automate model training and deployment, and keep track of experiments and results. In this tutorial, we have provided you with a practical example of how to use Azure Machine Learning and MLflow to train a CNN model for MNIST classification in PyTorch.

By following this tutorial, you should have gained a good understanding of how to use Azure Machine Learning and MLflow to streamline your machine learning workflow and achieve reproducible results. We encourage you to continue exploring the many capabilities of Azure Machine Learning and MLflow and to apply them to your own machine learning projects. In the coming weeks, we will be adding more tutorials and guides on Azure ML, so stay tuned for more!


