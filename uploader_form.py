from GoogleDrive.DriveConnections import DriveConnection
from ultralytics import YOLO
from cleanvision import Imagelab
import threading
import asyncio
import json
import glob
import time
import pprint
import sys
import os


image_path = "/home/lnv125/Desktop/petportrait_ai/uploader_form/images"
model_path = "/home/lnv125/Desktop/petportrait_ai/uploader_form/models"

def get_order_images_from_drive(order_number):
    """
    Fetches the order images from Google Drive.

    Returns:
    - True if successful.
    """
    # Download classification models
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
    isexist = os.path.exists(image_path)
    if not isexist:
        # Create a new directory because it does not exist
        os.makedirs(image_path)
        drive.download_files(
            folder_path=image_path, folder_id="1lNkSscwxEfq5WcgnSX4o53qjlZqTbc0v"
        )
    return True


def image_validation():
    """
    Validates the downloaded images for predefined issues.

    Returns:
    - A dictionary with validation results.
    """
    imagelab = Imagelab(data_path=image_path)

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
                    issue[column] = list(imagelab.issues[column].index)[i].split("/")[::-1][0]
    # print("{0} in file found at {1}".format(",".join(issue.keys()), issue.values()))
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
    model = YOLO(model_path+"/animal_breed_classification_yolov8.pt")
    total_images = glob.glob(image_path + "/*")
    results = model(total_images[0:2])  # predict on an image
    data = {}
    for image in results:
        result = image.probs
        name = image.names[result.top1]
        animal_type = name.split("_")[0]
        breed = "_".join(name.split("_")[1:])
        image_name = image.path.split("/")[::-1][0]
        data[image_name] = {"Type": animal_type, "Breed": breed}

    pprint.pprint(data)
    return data


async def find_pose_of_animal():
    """
    Identifies the pose of the animal in the images.
    """
    # Loading the model for animal pose classification
    model = YOLO(model_path+"/animal_pose_classification.pt")
    pose = {}
    total_images = glob.glob(image_path + "/*")
    results = model(total_images)
    for i in range(len(results)):
        result = results[i]
        box = result.boxes
        try:
            body = box.cls[0].item()
            pose[result.path] = {"face type:", result.names[body]}
        except Exception:
            pass

        try:
            face = box.cls[1].item()
            pose[result.path] = {"body type:", result.names[face]}
        except Exception:
            pass

    pprint.pprint(pose)
    return pose

# start = time.time()
# if len(sys.argv) != 4:
#     print("Usage: python uploader_form.py <order_number> <animal_type> <breed>")
#     sys.exit(1)

# order_number = sys.argv[1]
# animal_type = sys.argv[2]
# breed = sys.argv[3]


# get_order_images_from_drive(order_number)
# image_validation()
# identify_breed_of_animal()
# find_pose_of_animal()
# import shutil
# shutil.rmtree(image_path)
# stop = time.time()
# print(stop - start)
def run_async_in_thread(loop, coro):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def background_task():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(find_pose_of_animal())

def main(order_number):
    get_order_images_from_drive(order_number)
    validation_results = image_validation()
    breed_results = identify_breed_of_animal()

    # Start the background task
    thread = threading.Thread(target=background_task)
    thread.start()

    return {
        "validation_results": validation_results,
        "breed_results": breed_results
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python uploader_form.py <order_number>")
        sys.exit(1)

    order_number = sys.argv[1]
    # animal_type = sys.argv[2]  # You're not using this currently
    # breed = sys.argv[3]        # You're not using this currently

    results = main(order_number)
    print(json.dumps(results))
