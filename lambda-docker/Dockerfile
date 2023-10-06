# # python3.8 lambda base image
# # FROM public.ecr.aws/lambda/python:3.8
# FROM python:3

# RUN apt update && apt-get install ffmpeg libsm6 libxext6  -y

# #COPY ffmpeg-6.0-amd64-static /usr/local/bin/ffmpeg
# RUN pip install --upgrade pip
# RUN pip install ffmpeg

# RUN pip install numpy --no-cache-dir
# RUN pip install ultralytics --no-cache-dir
# RUN pip install awslambdaric --no-cache-dir

# # Copy function code to container
# COPY app.py ./

# # setting the CMD to your handler file_name.function_name
# CMD [ "app.handler" ]

FROM python:3.8

COPY handler.py ./
COPY models/animal_breed_classification_yolov8.pt ./

RUN mkdir /images/ ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install numpy
RUN pip install ultralytics
RUN pip install awslambdaric

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD ["handler.handler"]
