import requests
import base64
import numpy as np
from onnx import numpy_helper
import onnx
from google.protobuf.json_format import MessageToJson
import json
import tools.data_util as data_util
import tools.defaults as defaults

def main():
    """
    You need to run onnx runtime server on your localhost in advance.
    """
    ENDPOINT = 'http://localhost:8001/v1/models/default/versions/1:predict'
    json_request_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    colors = defaults.COLORS
    for color in colors:
        input_array = data_util.get_transformed_array(color)
        tensor_proto = numpy_helper.from_array(input_array)
        json_str = MessageToJson(tensor_proto, use_integers_for_enums=True)
        data = json.loads(json_str)

        inputs = {}
        inputs['input_0'] = data
        output_filters = ['output_0']

        payload = {}
        payload["inputs"] = inputs
        payload["outputFilter"] = output_filters

        res = requests.post(ENDPOINT, headers=json_request_headers, data=json.dumps(payload))
        raw_data = json.loads(res.text)['outputs']['output_0']['rawData']
        outputs = np.frombuffer(base64.b64decode(raw_data), dtype=np.float32)
        pred = np.argmax(outputs)

        assert defaults.COLORS[pred] == color
        print("Great! Label = {}, Pred = {}.".format(defaults.COLORS[pred], color))

if __name__ == '__main__':
    main()