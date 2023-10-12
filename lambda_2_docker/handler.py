from ultralytics import YOLO
import logging
import json
import base64
import boto3
import cv2
import numpy as np

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def find_pose_of_animal(image_list, jobid):
    """
    Identifies the pose of the animal in the images.
    """
    # Loading the YOLO model for animal pose classification from the given path
    model = YOLO("./animal_pose_classification.pt")
    pose = {}
    results = model(image_list)

    # Loop through each result and extract pose information
    for i in range(len(results)):
        result = results[i]
        box = result.boxes

        # Try extracting body type of the animal and add to the pose dictionary
        try:
            body = box.cls[0].item()
            # Try extracting face type of the animal and add to the pose dictionary
            face = box.cls[1].item()
            pose[f"image_{i}_{jobid}_.jpg"] = [{"face type":result.names[body]}, {"body type": result.names[face]}]
        except Exception:
            pass
    
    logger.info(pose)
    return pose


def handler(event, context):


    # Set up the S3 client
    s3: object = boto3.client("s3")
    bucket_name: str = "uploaderform-1"

    # Extract the key from the lambda event trigger
    key = event["Records"][0]["s3"]["object"]["key"]

    # Get the object (file) from the S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = response["Body"].read()
    data = json.loads(file_content) # Load the content of the file as JSON
    
    jobid = data["jobid"]

    image_list = []
    output_key = f"output/{jobid}/{jobid}_image_data.json"


    # Decode and process each image, and add it to the image_list
    for image in data["images"]:
        image_data = base64.b64decode(image)
        nparr = np.fromstring(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_list.append(img_np)
    pose_results = find_pose_of_animal(image_list, jobid)

    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket_name, output_key)
    s3_object.put(Body=(bytes(json.dumps(list(pose_results)).encode('UTF-8'))))

    return {"job_id":jobid,"breed_results": pose_results}