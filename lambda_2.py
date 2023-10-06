from ultralytics import YOLO
import logging
import pickle
import base64
import boto3
import glob
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def identify_breed_of_animal(model_path, image_path):
    """
    Identifies the breed of the animal in the images.

    Returns:
    - A dictionary with Type and Breed information.
    """
    # Loading the model for animal breed classification
    model = YOLO(model_path + "/animal_breed_classification_yolov8.pt")
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

    logger.info(data)
    return data


def lambda_handler(event, context):
    """
    AWS Lambda handler for classifying the breed of animals in images.

    The function does the following:
    1. Downloads a pickled file named 'image_data.pkl' from an S3 bucket. This file contains images
       encoded in base64 format.
    2. Decodes the base64 encoded images and saves them to a local directory.
    3. Uses a YOLO model to identify the breed of the animals in the images.

    Args:
        event (dict): AWS Lambda `event` object containing information about the triggering event.
        context (obj): AWS Lambda `context` object providing runtime information.

    Returns:
        dict: A dictionary with filenames as keys and their respective animal "Type" and "Breed" as values.
    """
    
    model_path = ""
    image_path = ""

    s3: object = boto3.client("s3")
    bucket_name: str = ""
    file_key = "image_data.pkl"

    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response["Body"].read()

    data = pickle.loads(file_content)

    isexist = os.path.exists(image_path)
    if not isexist:
        os.makedirs(image_path)
        for image_name, image_data in data.items():
            logger.info(image_name)
            image_data = base64.b64decode(image_data)
            with open(image_path + "/" + image_name, "wb") as image_file:
                image_file.write(image_data)

    breed_results = identify_breed_of_animal(model_path, image_path)
    return breed_results
