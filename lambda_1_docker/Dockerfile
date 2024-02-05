FROM python:3.10

COPY animal_breed_classification_yolov8.pt ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas==2.1.1
RUN pip install ultralytics==8.0.187
RUN pip install awslambdaric
RUN pip install python-dotenv==1.0.0
RUN pip install boto3==1.28.54

COPY handler.py ./

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD ["handler.handler"]
