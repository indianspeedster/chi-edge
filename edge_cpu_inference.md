::: {.cell .markdown}
# Using edge devices for CPU-based inference

:::


::: {.cell .markdown}

Machine learning models are most often trained in the "cloud", on powerful centralized servers with specialized resources (like GPU acceleration) for training machine learning models. 


However, for a variety of reasons including privacy, latency, and network connectivity or bandwidth constraints, it is often preferable to *use* these models (i.e. do inference) at "edge" devices located wherever the input data is/where the model's prediction is going to be used. 


These edge devices are less powerful and typically lack any special acceleration, so the inference time (the time from when the input is fed to the model, until the model outputs its prediction) may not be as fast as it would be on a cloud server - but we avoid having to send the input data to the cloud and then sending the prediction back.

:::

::: {.cell .markdown}

This notebook assumes you already have a "lease" available for a Raspberry Pi device on the CHI@Edge testbed. Then, it will show you how to:

* launch a "container" on that device
* attach an IP address to the container, so that you can access it over SSH
* transfer files to and from the container
* use a pre-trained image classification model to do inference on the edge device
* delete the container

:::

::: {.cell .markdown}
## Launch a container on an edge device

We will start by preparing our environment in this notebook, then launching a container on an edge device using our pre-existing lease.

:::

::: {.cell .markdown}

First, we load some required libraries:

:::

::: {.cell .code}
``` python
import chi
from chi import container
from chi import lease
import datetime
import os
```
:::

::: {.cell .markdown}

We indicate that we're going to use the CHI@Edge site. We also need to specify the name of the Chameleon "project" that this experiment is part of. The project name will have the format "CHI-XXXXXX", where the last part is a 6-digit number, and you can find it on your [user dashboard](https://chameleoncloud.org/user/dashboard/).

In the cell below, replace the project ID with your *own* project ID, then run the cell.

:::

::: {.cell .code}
``` python
chi.use_site("CHI@Edge")
chi.set("project_name", "CHI-XXXXXX")
```
:::

::: {.cell .markdown}

Next, we'll specify the lease ID. This notebook assumes you already have a "least" for a Raspberry Pi device on CHI@Edge. To get the ID of this lease,

* Vist the CHI@Edge ["reservations" page](https://chi.edge.chameleoncloud.org/project/leases/).
* Click on the lease name.
* On the following page, look for the value next to the word "Id" in the "Lease" section.

Fill in the lease ID inside the quotation marks in the following cell, then run the cell.


:::

::: {.cell .code}
``` python
lease_id ="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```
:::

::: {.cell .markdown}

Now, we are ready to launch a container!

-   **Container** : A container is like a logical "box" that holds everything needed to run an application. It includes the application itself, along with all the necessary prerequisite software, files, and settings it needs to work properly. 
-   **Image** : An image is like a pre-packaged "starting point" for a container. In this example, we're going to run a machine learning application written in Python, so we will use the `python:3.9-slim` image as a starting point for our container. This is a lightweight installation of the Debian Linux operating system with Python pre-installed.

When we create the container, we could also specify some additional arguments: 

-   `workdir`: the "working directory" - location in the container's filesystem from which any commands we specify will run.
-   `exposed_ports`: if we run any applications inside the container that need to accept incoming requests from a network, we will need to export a "port" number for those incoming requests. Any requests to that port number will be forwarded to this container.
-   `command`: if we want to execute a specific command immediately on starting the container, we can specify that as well.

but, we won't need to specify these for this particular experiment.

:::

::: {.cell .markdown}

First, we'll specify the name for our container - we'll include our username and the experiment name in the container name, so that it will be easy to identify our container in the CHI@Edge web interface.

:::

::: {.cell .code}
``` python
username = os.environ.get("USER")
expname = "edge-cpu"
# set a name for the container
# Note that underscore characters _ are not allowed - we replace each _ with a -
container_name = f"{username}-{expname}".replace("_","-")
```
:::

::: {.cell .markdown}

Then, we can create the container!

:::

::: {.cell .code}
``` python
try:
    my_container = container.create_container(
        container_name,
        image="python:3.9-slim",
        reservation_id=lease.get_device_reservation(lease_id),
        platform_version=2,
    )
except RuntimeError as ex:
    print(ex)
    print(f"Please stop and/or delete {container_name} and try again")
else:
    print(f"Successfully created container: {container_name}!")
```
:::


::: {.cell .markdown}

Once the container is created, you should be able to see it and monitor its status on the [CHI@Edge web interface](https://chi.edge.chameleoncloud.org/project/container/containers). (If there was any problem while creating the container, you can also forcefully delete the container from the interface, in order to be able to try again.)

:::


::: {.cell .markdown}
### Interacting with the container

Just like you ssh into a virtual machine and access that machine, you also can access the container by running terminal commands via container.execute() function.
:::

::: {.cell .code}
``` python
cmd = 'echo Hello'
print(cmd)

print(container.execute(my_container.uuid, cmd)["output"])
```

::: {.output .stream .stdout}
    echo Hello
    Hello
:::
:::

::: {.cell .markdown}
### Attaching a public ip address to the container

When you assign a public IP address, any exposed ports on your container can be reached over the public internet.
:::

::: {.cell .code}
``` python
public_ip = container.associate_floating_ip(my_container.uuid)

print(public_ip)
```

::: {.output .stream .stdout}
    129.114.34.182
:::
:::

::: {.cell .markdown}
### Transfering files to and from the container

-   To upload files to container we use `container.upload(container_ref: 'str', source: 'str', dest: 'str')` function.
-   to download files from container to our local we use `container.download(container_ref: 'str', source: 'str', dest: 'str')` function.
:::

::: {.cell .code}
``` python
container.upload(my_container.uuid, "./python_code", "/var/www/html")
#The code will be uploading some files which we will be going to use for our american sign language classification model
print("Files uploaded!")
```

::: {.output .stream .stdout}
    Files uploaded!
:::
:::

::: {.cell .markdown}
## Creating an image classification model using tflite

The folder which we previously uploaded contains:

-   model.py (The python file which contains all the code to run the model)
-   model.tflite (The tensorflow lite machine learning model for edge devices)
-   image.png (This image which is going to be used to make prediction)
-   Requirments.txt (Requirements file which is used to install all the requirements for our machine learning model)
:::

::: {.cell .markdown}
### Installing the required libraries

We will be installing some of the libraries that we are going to need for our ml model.
:::

::: {.cell .code}
``` python
cmd = "pip install -r requirements.txt"
print(cmd)
print(container.execute(my_container.uuid, cmd)["output"])
```

::: {.output .stream .stdout}
    pip install -r requirements.txt
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/site-packages (from -r requirements.txt (line 1)) (1.24.4)
    Requirement already satisfied: tflite-runtime in /usr/local/lib/python3.8/site-packages (from -r requirements.txt (line 2)) (2.13.0)
    Requirement already satisfied: pillow in /usr/local/lib/python3.8/site-packages (from -r requirements.txt (line 3)) (10.0.0)
:::
:::

::: {.cell .code}
``` python
cmd = "pip list"
print(cmd)
print(container.execute(my_container.uuid, cmd)["output"])
```

::: {.output .stream .stdout}
    pip list
    Package        Version
    -------------- -------
    numpy          1.24.4
    Pillow         10.0.0
    pip            23.0.1
    setuptools     57.5.0
    tflite-runtime 2.13.0
    wheel          0.40.0

    [notice] A new release of pip is available: 23.0.1 -> 23.1.2
    [notice] To update, run: pip install --upgrade pip
:::
:::

::: {.cell .markdown}
### Running the model
:::

::: {.cell .code}
``` python
cmd = "python model.py"
print(cmd)
print(container.execute(my_container.uuid, cmd)["output"])
```

::: {.output .stream .stdout}
    python model.py
    0.580392: fig
    0.568627: Granny Smith
    0.549020: spaghetti squash
:::
:::

::: {.cell .code}
``` python
```
:::
