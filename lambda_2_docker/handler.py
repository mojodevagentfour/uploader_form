from ultralytics import YOLO
import requests
import logging
import json
import base64
import boto3
import cv2
import numpy as np

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    AWS Lambda function handler for processing animal images to detect poses.

    Parameters:
    - event (dict): AWS Lambda event object containing details of the S3 object that triggered the function.
    - context (obj): AWS Lambda context object (not used in this function).

    Returns:
    - dict: A dictionary containing job id and the pose results.
    """

    # Set up the S3 client
    bucket_name = "uploaderform-1"
    s3_client = boto3.client("s3")

    # Extract the key from the lambda event trigger
    key = event["Records"][0]["s3"]["object"]["key"]

    # Get the object (file) from the S3 bucket
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    file_content = response["Body"].read()
    data = json.loads(file_content) # Load the content of the file as JSON
    jobid = data["jobid"]

    np_arr, image_name = [], []
    # Decode base64 images and convert them to numpy array format
    for index , base64_string in enumerate(data['images']):
        base64_file = base64_string["image_data"]
        file_name = base64_string["file_name"]
        image_bytes = base64.b64decode(base64_file)
        nparr = np.fromstring(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        np_arr.append(img_np)
        image_name.append(file_name)

    pose = {}
    model = YOLO("./animal_pose_classification.pt")
    results = model(np_arr)
    # Loop through each result and extract pose information
    for index, result in enumerate(results):
        box = result.boxes
        face, body = "", ""
        l= box.cls.tolist()
        l.sort()
        box_m = box.xyxy.tolist()
        box_m.sort()
        _, img_encoded = cv2.imencode('.jpg', np_arr[index])

        # Convert the image to base64
        base64_encoded_image = base64.b64encode(img_encoded).decode('utf-8')
        try:
            face = result.names[l[0]]
            # Try extracting face type of the animal and add to the pose dictionary
            body = result.names[l[1]]
        
            # Iterate through the bounding boxes
            x1, y1, x2, y2 = box_m[0]
            # print(x1, y1, x2, y2)

            # img = cv2.imread(image_list[index])
            # Crop the object using the bounding box coordinates
            ultralytics_crop_object = np_arr[index][int(y1):int(y2), int(x1):int(x2)]
            
            # Save the cropped object as an image
            name = result.path.split(".")[0]
            dummy_image = ultralytics_crop_object
            encoded_image = cv2.imencode('.png', dummy_image)[1].tobytes()

            # Convert the image to base64
            base64_encoded_image = base64.b64encode(encoded_image).decode('utf-8')

            # Now 'base64_encoded_image' contains the base64 representation of the image
        except Exception as e:
            pass
        pose[image_name[index]] = [face, body]
        data["images"][index]['image_data'] = base64_encoded_image
        data["images"][index]["animal_pose"] = pose[image_name[index]]

    output_key = f"output/{jobid}/{jobid}_image_data.json"
    s3_resource = boto3.resource('s3')
    s3_object = s3_resource.Object(bucket_name, output_key)
    s3_object.put(Body=json.dumps(data))
    
    
    # api_url = "http://104.225.221.251:8000/start/"

    # payload = json.dumps({
    # "order_number": f"{jobid}"
    # })
    # headers = {
    # 'Content-Type': 'application/json'
    # }

    # try:
    #     response = requests.post(api_url, headers=headers, data=payload, timeout=0.5)
    #     print(response.text)
    # except requests.RequestException as e:
    #     pass

    return {"job_id": jobid, "breed_results": pose}
