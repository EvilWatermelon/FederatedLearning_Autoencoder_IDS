# if-and-auto: A Flower / PyTorch app

## Running the Model on PC / Singular Device

Go to the Folder 'Python_FL_Model' and follow the 'README'

## Running the Model on Embedded Devices (Rasberry Pis etc.)

Follow this Tutorial https://github.com/adap/flower/tree/main/examples/embedded-devices. 
Instead of cloning their example use mine.

## Launching the Flower SuperLink

On your development machine, launch the 'SuperLink'. You will connnect Flower 'SuperNodes' to it in the next step.
```bash
flower-superlink --insecure
```
##Connecting Flower SuperNodes

With the 'SuperLink' up and running, we can now launch a 'SuperNode' on each embedded device. To do this, make sure you know the IP address of the machine running the 'SuperLink' and that the necessary data has been copied to the device.
Ensure the Python environment you created earlier when setting up your device has all dependencies installed. 
Now, launch your 'SuperNode'  
```bash
# Repeat for each embedded device (adjust SuperLink IP and partition-id)
flower-supernode  --insecure      --superlink SuperLink IP:9092    --node-config "partition-id=0 num-partitions=4"
```
Repeat for each embedded device that you want to connect to the 'SuperLink'.

## Run the Flower App

With both the long-running server ('SuperLink') and two 'SuperNodes' up and running, we can now start run. Let's first update the Flower Configuration file to add a new 'SuperLink' connection.

Locate your Flower configuration file by running:
```bash
flwr config list
```
```bash
# Example output:
Flower Config file: /path/to/your/.flwr/config.toml
SuperLink connections:
 supergrid
 local (default)
```
Open this configuration file and add a new 'SuperLink' connection at the end:
```bash
[superlink.embedded-federation]
address = "127.0.0.1:9093" # ControlAPI of your SUPERLINK
insecure = true
```
Finally, run your Flower App in your federation:
```bash
flwr run . embedded-federation
```
