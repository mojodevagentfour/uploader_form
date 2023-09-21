from dagster import job, resource, op
from GoogleDrive.DriveConnections import DriveConnection
import paramiko
from tensordock import TensorDock
import sys
import os
import json
import time


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
def run_uploader_form(context, cread:dict):
    data = context.resources.order_data
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ip = cread["Ip"]
    order_number = data["order_number"]
    # animal_type = data["animal_type"]
    # breed = data["breed"]
    os.system(f"ssh-keygen -f '/home/lnv125/.ssh/known_hosts' -R {ip}")
    os.system("fuser -k 22/tcp")
    time.sleep(15)
    # command = "sudo apt install python3-pip -y; \
    #             pip3 install ultralytics; \
    #             pip3 install cleanvision;"
    command = f"sudo docker run -it mojocreator/uploader_form:0.1.1 python uploader_form.py {order_number}"

    try:
        ssh.connect(
            ip,
            username="user",
            password="test@1pass",
        )
        result_lines = []
        _, stdout, __ = ssh.exec_command(command, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")
            result_lines.append(line.strip())
            if '{"validation_results":' in line:
                results = json.loads(result_lines[-1])
                break

    except paramiko.SSHException as error:
        print(error)
        run_uploader_form(context, cread)
    ssh.close()
    return results



@op
def destroying_instance(results:dict, cread:dict):
    """
    Destroys the server instances once operations are complete.
    """
    server_id = cread["Id"]
    status = TensorDock().delete_server(server_id=server_id)
    if status["success"]:
        print("Instance destroyed..!!!")
    else:
        print("Instance has not been destroyed...!!!")


@job(resource_defs={"order_data": order_data_resource})
def generateuploader_form():
    """
    Main job pipeline that orchestrates the order processing.
    """
    cread = running_state_instance(start_instances())
    results = run_uploader_form(cread)
    destroying_instance(results, cread)
