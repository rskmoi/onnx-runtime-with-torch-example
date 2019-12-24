import grpc
from protobuf import prediction_service_pb2_grpc
from protobuf import predict_pb2, onnx_ml_pb2
import numpy as np
import tools.data_util as data_util
import tools.defaults as defaults
from onnx import numpy_helper


def main():
    """
    You need to run onnx runtime server on your localhost in advance.
    """
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        for color in defaults.COLORS:
            input_array = data_util.get_transformed_array(color)
            onnx_tensor_proto = numpy_helper.from_array(input_array)
            tensor_proto = onnx_ml_pb2.TensorProto()
            tensor_proto.ParseFromString(onnx_tensor_proto.SerializeToString())
            inputs = {"input_0": tensor_proto}

            request = predict_pb2.PredictRequest(inputs=inputs)
            predict = stub.Predict(request)
            output = predict.outputs["output_0"]
            output_array = numpy_helper.to_array(output)
            assert defaults.COLORS[np.argmax(output_array)] == color
            print("Great! Label = {}, Pred = {}.".format(defaults.COLORS[np.argmax(output_array)], color))


if __name__ == '__main__':
    main()