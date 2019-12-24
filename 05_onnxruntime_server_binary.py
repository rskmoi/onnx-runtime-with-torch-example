import requests
import numpy as np
from onnx import numpy_helper
import tools.data_util as data_util
import tools.defaults as defaults

from protobuf import predict_pb2, onnx_ml_pb2


def main():
    """
    You need to run onnx runtime server on your localhost in advance.
    """
    ENDPOINT = 'http://localhost:8001/v1/models/default/versions/1:predict'
    json_request_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    pb_request_headers = {
        'Content-Type': 'application/octet-stream',
        'Accept': 'application/octet-stream'
    }


    colors = defaults.COLORS
    for color in colors:
        input_array = data_util.get_transformed_array(color)
        onnx_tensor_proto = numpy_helper.from_array(input_array)
        tensor_proto = onnx_ml_pb2.TensorProto()
        tensor_proto.ParseFromString(onnx_tensor_proto.SerializeToString())

        predict_request = predict_pb2.PredictRequest()
        predict_request.inputs['input_0'].CopyFrom(tensor_proto)
        predict_request.output_filter.append('output_0')

        payload = predict_request.SerializeToString()
        res = requests.post(ENDPOINT, headers=pb_request_headers, data=payload)
        actual_result = predict_pb2.PredictResponse()
        actual_result.ParseFromString(res.content)
        outputs = numpy_helper.to_array(actual_result.outputs['output_0'])
        pred = np.argmax(outputs)
        assert defaults.COLORS[pred] == color
        print("Great! Label = {}, Pred = {}.".format(defaults.COLORS[pred], color))


if __name__ == '__main__':
    main()