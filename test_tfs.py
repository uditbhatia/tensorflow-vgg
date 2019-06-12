
import json
import utils
import requests


test_images = ["./test_data/tiger.jpeg","./test_data/puzzle.jpeg" ]
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:2]})
print('Data: {} ... {}'.format(data, data[len(data)-52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/vgg197:predict', data=data, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']

print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(predictions[0])], test_labels[0], class_names[np.argmax(predictions[0])], test_labels[0]))