FROM python:3.8

COPY animal_breed_classification_yolov8.pt ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install numpy
RUN pip install ultralytics
RUN pip install awslambdaric
RUN pip install python-dotenv
RUN pip install boto3

COPY handler.py ./

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD ["handler.handler"]
