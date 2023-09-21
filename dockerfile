FROM python:3.8.3

COPY cred.json /root/
COPY uploader_form.py /root/

RUN python -m pip install --upgrade pip; \
    apt-get update; \
    apt-get install -y libunwind-dev \
    apt-get update; \
    pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib; \
    pip install oauth2client; \
    apt --fix-broken install -y; \
    apt-get remove --purge aria2 -y; \
    apt-get autoremove -y; \
    apt-get install aria2 -y; \
    apt --fix-broken install;
    
WORKDIR /root

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN apt-get update; \
    apt-get remove -y google-perftools; \
    apt-get update; \ 
    apt-get install -y libunwind8-dev libunwind8; \
    apt-get install google-auth-oauthlib==0.4.1; \
    apt --fix-broken install -y;

RUN pip install ultralytics; \ 
    pip install cleanvision;
    
RUN mkdir GoogleDrive
ADD GoogleDrive/* /root/GoogleDrive/

RUN mkdir models
ADD models/* /root/models/

