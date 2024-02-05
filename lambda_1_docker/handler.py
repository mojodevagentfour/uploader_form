import json
import base64
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import logging
import boto3

# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    AWS Lambda function handler to process images and identify animal type and breed.

    Parameters:
    - event (dict): AWS Lambda event object. It is expected to contain:
        * 'jobid' (str): The job identifier.
        * 'images' (list): List containing base64 encoded images to process. 
            Each item in the list is a dictionary with 'image_data' key containing base64 encoded image data.
    - context (obj): AWS Lambda context object (not used in this function).

    Returns:
    - dict: A dictionary containing:
        * 'job_id' (str): The job identifier.
        * 'breed_results' (dict): Dictionary with image names as keys and detected animal type and breed as values.

    Workflow:
    1. Fetches the image data from the 'event'.
    2. Decodes the base64 encoded image and converts it into a numpy array format suitable for processing.
    3. Uses the YOLOv8 model to predict animal type and breed.
    4. Stores the predictions along with the image names in the results dictionary.
    5. Writes the results to an S3 bucket.
    6. Returns the job id and the results.
    """
   
    bucket_name: str = "uploaderform-1" # Bucket where the results will be stored
    key = f"input/{event['jobid']}/{event['jobid']}_image_data.json" # Location (key) in the bucket where results will be stored
    model = YOLO("./animal_breed_classification_yolov8.pt") # Load the YOLO model for animal type and breed detection

    np_arr, image_name = [], []
    # Decode base64 images and convert them to numpy array format
    for index , base64_string in enumerate(event['images']):
        base64_file = base64_string["image_data"]
        file_name = base64_string["file_name"]
        image_bytes = base64.b64decode(base64_file)
        nparr = np.fromstring(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        np_arr.append(img_np)
        image_name.append(file_name)

    # Predict animal type and breed using the YOLO model
    results = model.predict(np_arr)  
    data = {}
    # # Extract the predictions from the results
    for index, image in enumerate(results):
        result = image.probs
        name = image.names[result.top1]
        animal_type = name.split("_")[0]
        breed = "_".join(name.split("_")[1:])
        data[image_name[index]] = {"Type": animal_type, "Breed": breed}
        event['images'][index]["animal_type"] = animal_type
        event['images'][index]["breed"] = breed

    # Save the results to the specified S3 bucket
    s3 = boto3.resource('s3')
    s3object = s3.Object(bucket_name, key)
    s3object.put(Body=json.dumps((event)))

    # Return the results 
    return {"job_id":event['jobid'],"breed_results": data}
