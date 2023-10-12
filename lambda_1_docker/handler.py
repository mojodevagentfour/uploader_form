import json
import base64
from ultralytics import YOLO
import cv2
import numpy as np
import glob
import logging
import boto3


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    AWS Lambda handler that performs the following tasks:
    1. Retrieves a image data
    2. Extracts all files from the data and re-uploads them into to the S3 bucket.
    3. Converts images found in a specified path to base64 format.
    4. Reads a pickled data from a file named 'image_data.pkl'.
    5. Validates images using the `image_validation` function.
    6. Classifies the animal breed in the images using the `identify_breed_of_animal` function.

    Args:
        event (dict): Contains information about the triggering event.
        context (LambdaContext): AWS Lambda context object which provides metadata on the currently executing lambda function.

    Raises:
        e (Exception): Propagates exceptions raised during S3 operations or image processing.

    Returns:
        dict: A dictionary containing validation results and breed identification results.
    """
    bucket_name: str = "uploaderform-1"
    key = f"input/{event['jobid']}/{event['jobid']}_image_data.json"
    model = YOLO("./animal_breed_classification_yolov8.pt")

    np_arr = []
    for index , base64_string in enumerate(event['images'][:2]):
      base64_file = base64_string
      image_bytes = base64.b64decode(base64_file)
      nparr = np.fromstring(image_bytes, np.uint8)
      img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      np_arr.append(img_np)
    
    results = model(np_arr) 
    
    data = {}
    for image in results:
        result = image.probs
        name = image.names[result.top1]
        animal_type = name.split("_")[0]
        breed = "_".join(name.split("_")[1:])
        image_name = image.path.split("/")[::-1][0]
        data[image_name] = {"Type": animal_type, "Breed": breed}


    s3 = boto3.resource('s3')

    s3object = s3.Object(bucket_name, key)

    s3object.put(
        Body=(bytes(json.dumps(event).encode('UTF-8')))
    )


    return {"job_id":event['jobid'],"breed_results": data}
