import subprocess
import requests
import os
import json
import time

# TensorDock Core Cloud API Documentation: https://documenter.getpostman.com/view/10732984/UVC3j7Kz#intro

class TensorDock:
    def __init__(self) -> None:
        # List to store the instances deployed
        self.instances = []
        # Dictionary to track the running instances
        self.running = {}

    def deploy_server(self):
        """
        Deploys servers on TensorDock based on the count provided.

        Parameters:
        - count (int): Number of servers to be deployed

        Returns:
        - list: List of instance details
        """
        deploy_url = "https://console.tensordock.com/api/deploy/single/custom"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        deploy_payload = {
            "api_key": "recabQkXrkbei2crf",
            "api_token": "FCACNCSdkGHeVJOommokqIJlNnCOFO",
            "admin_user": "user",
            "admin_pass": "test@1pass",
            "instance_type": "cpu",
            "cpu_model": "INTEL_XEON_SCALABLE",
            "cpu_count": 1,
            "vcpus": 1,
            "ram": 4,
            "storage": 80,
            "storage_class": "io1",
            "os": "Ubuntu 22.04 LTS",
            "location": "na-us-chi-1",
            "name": "ray_worker",
            """cloud_init""": """runcmd:
            - docker login -u mojocreator -p dckr_pat_gYXAFOgplKc7Rgrz5p8Qk2KsHhE
            - docker run --name=ray_worker --gpus=all --restart=always --network=host -it -d mojocreator/pet-potrait:0.1.9""",
        }
        deploy_req = requests.post(deploy_url, headers=headers, data=deploy_payload)
        self.id = deploy_req.json()["server"]["id"]
        self.ip = deploy_req.json()["server"]["ip"]
        self.status = deploy_req.json()["success"]
        return self.id, self.ip, self.status

    def wait_until_deployed(self, _id, _ip):
        """
        Continuously checks if the server is in a running state.
        """

        self.id = _id
        self.ip = _ip
        print(self.id, self.ip, "+++====+++===")

        url = "https://console.tensordock.com/api/deploy/status"

        payload = {
            "api_key": "recabQkXrkbei2crf",
            "api_token": "FCACNCSdkGHeVJOommokqIJlNnCOFO",
            "server": self.id,
        }

        response = requests.request("POST", url, data=payload)
        print(response.json(), "+==+++++===+++")
        if response.json()["status"] == "Success Or Started":
            self.running[self.id] = "success"
        else:
            time.sleep(15)
            self.wait_until_deployed(self.id, self.ip)

        return self.running

    def delete_server(self, server_id):
        """
        Deletes the servers that are successfully running.
        """
        if list(server_id.values())[0] == "success":
            url = f"https://console.tensordock.com/api/delete/single?api_key=recabQkXrkbei2crf&api_token=FCACNCSdkGHeVJOommokqIJlNnCOFO&server={list(server_id.keys())[0]}"
            response = requests.request("GET", url=url)
            return response.json()
