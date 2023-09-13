from ultralytics import YOLO
from GoogleDrive.DriveConnections import DriveConnection
from ultralytics import YOLO
from cleanvision import Imagelab
import glob
import time
import pprint
import sys
import os


def get_order_images_from_drive(order_number):
    """
    Fetches the order images from Google Drive.

    Returns:
    - True if successful.
    """
    # Download classification models
    model_path = os.getcwd()+"/models/"
    isexist = os.path.exists(model_path)
    if not isexist:
        # Create a new directory because it does not exist
        os.makedirs(model_path)
    drive = DriveConnection(order_number="models")
    drive.download_files(
        folder_path=model_path, folder_id="1kZQr8st_CMjoTbm1evJPvdwVPlKAdUiY"
    )

    # download Images
    print(order_number, "+=++++====++++==")
    drive = DriveConnection(order_number=order_number)
    print("Will check for orders in Drive")
    path = os.getcwd()+"/images/"
    isexist = os.path.exists(path)
    if not isexist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    drive.download_files(
        folder_path=path, folder_id="1lNkSscwxEfq5WcgnSX4o53qjlZqTbc0v"
    )
    time.sleep(10)
    return True


def image_validation():
    """
    Validates the downloaded images for predefined issues.

    Returns:
    - A dictionary with validation results.
    """
    imagelab = Imagelab(data_path=os.getcwd()+"/images/")

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


def identify_breed_of_animal():
    """
    Identifies the breed of the animal in the images.

    Returns:
    - A dictionary with Type and Breed information.
    """
    # Loading the model for animal breed classification
    model = YOLO(os.getcwd()+"/models/animal_breed_classification_yolov8.pt")

    image_folder_path = os.getcwd()+"/images"
    data = {}

    total_images = glob.glob(image_folder_path + "/*.jpeg")
    for image in total_images[0:2]:
        results = model(image)  # predict on an image
        result = results[0].probs
        name = results[0].names[result.top1]
        animal_type = name.split("_")[0]
        breed = "_".join(name.split("_")[1:])
        data[image] = {"Type": animal_type, "Breed": breed}

    pprint.pprint(data)
    return data


def find_pose_of_animal():
    """
    Identifies the pose of the animal in the images.
    """
    # Loading the model for animal pose classification
    model = YOLO(os.getcwd()+"/models/animal_pose_classification.pt")
    image_folder_path = os.getcwd()+"/images"

    total_images = glob.glob(image_folder_path + "/*.jpeg")
    for image in total_images:
        results = model(image)
        result = results[0]
        box = result.boxes

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


if len(sys.argv) != 4:
    print("Usage: python uploader_form.py <order_number> <animal_type> <breed>")
    sys.exit(1)

order_number = sys.argv[1]
animal_type = sys.argv[2]
breed = sys.argv[3]


get_order_images_from_drive(order_number)
image_validation()
identify_breed_of_animal()
find_pose_of_animal()
