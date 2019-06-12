docker run -p 8501:8501 --mount type=bind,source=/tmp/model/vgg199,target=/models/vgg199 -e MODEL_NAME=vgg199 -t tensorflow/serving



