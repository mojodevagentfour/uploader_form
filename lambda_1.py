from cleanvision import Imagelab
from ultralytics import YOLO
from io import BytesIO
import io
import zipfile
import logging
import zipfile
import base64
import boto3
import glob
import pickle


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def image_validation(image_path):
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
    logger.info(issue)
    if issue:
        return {"validation_result": issue}
    else:
        return {"validation_result": True}


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
    AWS Lambda handler that performs the following tasks:
    1. Retrieves a zip file from an S3 bucket.
    2. Extracts all files from the zip and re-uploads them individually to the S3 bucket.
    3. Converts images found in a specified path to base64 format.
    4. Reads a pickled data from a file named 'dictionary_data.pkl'.
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
    model_path = "models"
    image_path = "images"
    jobid = event["jobid"]
    bucket_name: str = "uploaderform-1"
    key = "image_data.pkl"

    # decode images in base64 format.
    for index , image in enumerate(event['images']):
        image_data = base64.b64decode(image)
        with open(image_path + "/" + f"image_{index}.jpg", "wb") as image_file:
            image_file.write(image_data)

    serialized_data = pickle.dumps(event)
    session = boto3.Session()

    s3 = session.resource('s3')
    s3.Object(bucket_name, key=f"{jobid}/{key}").put(Body=serialized_data)
    print('Write in S3 successful.')
    # run the clean vision and animal type and breed classification.
    validation_results = image_validation(image_path)
    breed_results = identify_breed_of_animal(model_path, image_path)

    return {"validation_results": validation_results, "breed_results": breed_results}

# open a file, where you stored the pickled data
file = open('image_data.pkl', 'rb')

# dump information to that file
image_list = pickle.load(file)

# close the file
file.close()
print(lambda_handler(event=image_list, context=''))