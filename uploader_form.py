from ultralytics import YOLO
from cleanvision import Imagelab
import threading
import asyncio
import json
import glob
import pprint
import os
import base64
import pickle


image_path = "/home/lnv125/Desktop/petportrait_ai/aws_infra/images"
model_path = "/home/lnv125/Desktop/petportrait_ai/aws_infra/models"

def get_order_images_from_drive(data):
    """
    Fetches the order images from Google Drive.

    Returns:
    - True if successful.
    """
    isexist = os.path.exists(image_path)
    if not isexist:
        os.makedirs(image_path)
        for image_name, image_data in data.items():
            print(image_name,"+++===++++")
            image_data = base64.b64decode(image_data)
            with open(image_path+"/"+image_name, "wb") as image_file:
                image_file.write(image_data)

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
    # Loop through each result and extract pose information
    for i in range(len(results)):
        result = results[i]
        box = result.boxes

        # Try extracting body type of the animal and add to the pose dictionary
        try:
            body = box.cls[0].item()
            # Try extracting face type of the animal and add to the pose dictionary
            face = box.cls[1].item()
            pose[f"image_{i}.jpg"] = [{"face type":result.names[body]}, {"body type": result.names[face]}]
        except Exception:
            pass

    pprint.pprint(pose)
    return pose

def run_async_in_thread(loop, coro):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def background_task():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(find_pose_of_animal())

def main(data):
    get_order_images_from_drive(data)
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
    
    file = open('dictionary_data.pkl', 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    results = main(data)
    print(json.dumps(results))
