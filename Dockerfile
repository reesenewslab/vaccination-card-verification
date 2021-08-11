FROM ubuntu:18.04

RUN apt-get update \    
    && apt-get install tesseract-ocr -y \
    python3 \    
    python3-pip \    
    && apt-get clean \
    && apt-get autoremove

RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD . /home/App
WORKDIR /home/App
COPY requirements.txt ./
COPY . .

RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt

# VOLUME ["/data"]
# EXPOSE 5000 5000
# CMD ["python3","OCRRun.py"]
# python verify_card.py --template templates/CDC_card_template_01.png --image input.jpg
# CMD ["python3","verify_card.py","--template","templates/CDC_card_template_01.png","--image","input.jpg"]