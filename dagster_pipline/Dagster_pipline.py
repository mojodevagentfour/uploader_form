from dagster import job, resource, op
from GoogleDrive.DriveConnections import DriveConnection
import paramiko
from ultralytics import YOLO
from cleanvision import Imagelab
from tensordock import TensorDock
import sys
import os
import glob
import time
import pprint


@resource
def order_data_resource(init_context):
    """
    Resource definition for the order data.
    """
    return init_context.resource_config


@op
def start_instances():
    """
    Starts instances on TensorDock.

    Returns:
    - Dictionary with _id and _ip information.
    """
    _id, _ip, status = TensorDock().deploy_server()
    if status:
        return {"_id": _id, "_ip": _ip}
    else:
        sys.exit()


@op
def running_state_instance(start_instances_output: dict):
    """
    Monitors the state of the instances and ensures they're running.

    Returns:
    - Instance running status.
    """
    _id = start_instances_output["_id"]
    _ip = start_instances_output["_ip"]
    running = TensorDock().wait_until_deployed(_id, _ip)
    if list(running.values())[0] == "success":
        return {"Ip": _ip, "Id": _id}
    else:
        exit()


@op(required_resource_keys={"order_data"})
def run_uploader_form(context, cread: dict):
    data = context.resources.order_data
    print(data, "+=+++===++++")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ip = cread["Ip"]
    order_number = data["order_number"]
    animal_type = data["animal_type"]
    breed = data["breed"]
    os.system(f"ssh-keygen -f '/home/lnv125/.ssh/known_hosts' -R {ip}")
    command = f"python3 /home/user/uploader_form/uploader_form.py {order_number} {animal_type} {breed}"

    try:
        ssh.connect(
            ip,
            username="user",
            password="test@1pass",
        )

        _, stdout, __ = ssh.exec_command(command, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")

        return cread

    except paramiko.SSHException as error:
        print(error)


@op
def destroying_instance(cread: dict):
    """
    Destroys the server instances once operations are complete.
    """
    server_id = cread["Id"]
    print(server_id, "++++====++++")
    status = TensorDock().delete_server(server_id=server_id)
    if status["success"]:
        print("Instance destroyed..!!!")
    else:
        print("Instance has not been destroyed...!!!")


@job(resource_defs={"order_data": order_data_resource})
def generatekohya_lora():
    """
    Main job pipeline that orchestrates the order processing.
    """
    # destroying_instance(
    run_uploader_form(running_state_instance(start_instances()))#)
