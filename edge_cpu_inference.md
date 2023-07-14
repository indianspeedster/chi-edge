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
## Getting Started
:::

::: {.cell .markdown}
### Loading the Required Libraries
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
### Setting up some of the environment variables
:::

::: {.cell .code}
``` python
# Before we go any further, we need to select which Chameleon site we will be using.
chi.use_site("CHI@Edge")
#You can find your project ID on the user dashboard here: https://chameleoncloud.org/user/dashboard/
chi.set("project_name", "CHI-231095")
```

::: {.output .stream .stdout}
    Now using CHI@Edge:
    URL: https://chi.edge.chameleoncloud.org
    Location: University of Chicago, Chicago, Illinois, USA
    Support contact: help@chameleoncloud.org
:::
:::

::: {.cell .markdown}
### Creating a lease for the edge device

View the host calendar to check availability <https://chi.edge.chameleoncloud.org/project/leases/calendar/device/>
:::

::: {.cell .code}
``` python
# get your username, this will be used to make the lease identifiable for you.
username = os.environ.get("USER")

# machine name refers to the "type" of device
device_name = "iot-rpi-cm4-02"

# these are start and end dates for the lease
start_date, end_date = lease.lease_duration(days=2)
exp_start_time = datetime.datetime.now().strftime("%Y%_m_%d_%H_%M_%S")

lease_name = f"{username}-{device_name}-{exp_start_time}"

reservations = []
lease.add_device_reservation(reservations, count=1, device_name = device_name )
container_lease = lease.create_lease(lease_name, reservations)
lease_id = container_lease["id"]

print(f"Reservation made with name {lease_name} and uuid {lease_id}")
print("waiting for lease to start. This usually takes close to a minute.")
lease.wait_for_active(lease_id)
print("Lease successfully created")
```
:::

::: {.cell .markdown}
### Launching the first Container

-   **Container** : A container is like a virtual box that holds everything needed to run a computer program. It includes the program itself, along with all the necessary files and settings it needs to work properly. Containers make it easy to move programs from one computer to another without any problems. Containers can be easily created and destroyed which we will see in coming steps

-   **Image** : image is a self-contained package that contains all the necessary components to create and run a containerized program. It simplifies the process of sharing and running applications by bundling them into a single image that can be easily distributed and executed on different computers. Here in the below code we used an Image named *python:3.9-slim* which is a light weight version of python and this image will make our container to run any of the python program easily.

There are further some of the arguments that has been used in the code like:

-   workdir : it is used to set the working directory of the container.
-   exposed_ports : To expose a given port of the container
-   command : If you want to make sure that whenever the container is built you shuld run a specific command, you can use this.
:::

::: {.cell .code}
``` python

lease_id ="bf826fad-66b5-4eab-b563-8823b7e1d50a"
print("Creating container ...")
username = os.environ.get("USER")
device_name = "iot-rpi-cm4-02"
# set a name for the container. Becaue CHI@Edge uses Kubernetes, ensure that underscores aren't in the name
container_name = f"{username}-{device_name}-ml-app".replace("_","-")

try:
    my_container = container.create_container(
        container_name,
        image="python:3.8-slim",
        command=["python", "-m", "http.server", "8000"],
        workdir="/var/www/html",
        exposed_ports=[8000],
        reservation_id=lease.get_device_reservation(lease_id),
        platform_version=2,
    )
except RuntimeError as ex:
    print(ex)
    print(f"please stop and/or delete {container_name} and try again")
else:
    print(f"Successfully created container: {container_name}!")
```

::: {.output .stream .stdout}
    Creating container ...
    Successfully created container: cp3793-nyu-edu-iot-rpi-cm4-02-webserver!
:::
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
