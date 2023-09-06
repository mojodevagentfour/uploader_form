from dagster import job, resource, op
from GoogleDrive.DriveConnections import DriveConnection
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


@op(required_resource_keys={"order_data"})
def get_order_images_from_drive(context):
    """
    Fetches the order images from Google Drive.

    Returns:
    - True if successful.
    """
    #Download classification models
    model_path = "../models/"
    isexist = os.path.exists(model_path)
    if not isexist:
        # Create a new directory because it does not exist
        os.makedirs(model_path)
    drive = DriveConnection(order_number="models")
    drive.download_files(folder_path=model_path, folder_id='1kZQr8st_CMjoTbm1evJPvdwVPlKAdUiY')

    #download Images
    data = context.resources.order_data
    print(data, "+=++++====++++==")
    drive = DriveConnection(order_number=data["order_number"])
    print("Will check for orders in Drive")
    path = "../images/"
    isexist = os.path.exists(path)
    if not isexist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    drive.download_files(folder_path=path, folder_id='1lNkSscwxEfq5WcgnSX4o53qjlZqTbc0v')
    time.sleep(10)
    return True


@op
def image_validation(drive_status):
    """
    Validates the downloaded images for predefined issues.

    Returns:
    - A dictionary with validation results.
    """
    imagelab = Imagelab(data_path="../images/")

    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()

    # imagelab.report()
    # Produce a neat report of the issues found in your dataset
    issue = {}
    count = 0
    for column in imagelab.issues.columns:
        if "_issue" in column:
            for i in range(len(imagelab.issues)):
                if imagelab.issues[column][i]:
                    issue[count] = "{} in file found at {}".format(
                        str(column).upper(),
                        list(imagelab.issues[column].index)[i].split("/")[::-1][0],
                    )
                    count += 1
                    issue[column] = list(imagelab.issues[column].index)[i].split("/")[
                        ::-1
                    ][0]
    print("{0} in file found at {1}".format(",".join(issue.keys()), issue.values()))

    print()
    # Logic to validate the image...
    validation_result = "{Defecting reason} in file found at {file name}"  # Example result; replace with your actual logic
    if issue:
        return {"validation_result": issue}
    else:
        return {"validation_result": True}


@op
def identify_breed_of_animal(validation_result: dict):
    """
    Identifies the breed of the animal in the images.

    Returns:
    - A dictionary with Type and Breed information.
    """
    # Loading the model for animal breed classification
    model = YOLO(
        "../models/animal_breed_classification_yolov8.pt"
    )  # load a pretrained model (recommended for training)
    image_folder_path = (
        "../images"  # Replace "/content/image_files" into your folder path
    )

    # Example usage:
    data = {}
    total_images = glob.glob(image_folder_path + "/*.jpeg")
    for image in total_images:
        # Use the model
        # print(image)
        results = model(image)  # predict on an image
        result = results[0].probs
        name = results[0].names[result.top1]
        animal_type = name.split("_")[0]
        breed = "_".join(name.split("_")[1:])
        data[image] = {"Type": animal_type, "Breed": breed}
    pprint.pprint(data)
    return data


@op
def find_pose_of_animal(data):
    """
    Identifies the pose of the animal in the images.
    """
    # Loading the model for animal pose classification
    model = YOLO(
        "../models/animal_pose_classification.pt"
    )  # load a pretrained model (recommended for training)
    image_folder_path = (
        "../images"  # Replace "/content/image_files" into your folder path
    )

    # Example usage:
    total_images = glob.glob(image_folder_path + "/*.jpeg")
    for image in total_images:
        # Use the model
        results = model(image)
        result = results[0]
        box = result.boxes
        # cords = box.xyxy.tolist()
        try:
            body = box.cls[0].item()
            print("face type:", result.names[body])
        except Exception:
            pass
        try:
            face = box.cls[1].item()
            print("body type:", result.names[face])
        except Exception:
            pass
        # conf = box.conf.item()
        # print("Object type:", result.names[class_id])
        # print("Coordinates:", cords)
        # print("Probability:", conf)
        # res_plotted = results[0].plot()


@op
def start_instances(data):
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
    return running


@op
def destroying_instance(server_id: dict):
    """
    Destroys the server instances once operations are complete.
    """
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
    destroying_instance(
        running_state_instance(
            start_instances(
                find_pose_of_animal(
                    identify_breed_of_animal(
                        image_validation(get_order_images_from_drive())
                    )
                )
            )
        )
    )
