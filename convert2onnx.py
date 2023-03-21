import numpy as np
import torch
from torch import nn
import torch.onnx
import onnx
import onnxruntime

batch_size = 1
channles = 3
opset_version = 10

simple_model_name = "./data/simple_model.onnx"
multi_model_name = "./data/multi_model.onnx"


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class MultiInputConv(nn.Module):
    def __init__(self):
        super(MultiInputConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.features(x)
        return x


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


########### simple model ##########

simple_model = SimpleConv()
simple_model.eval()

# Input to the model
x = torch.randn(batch_size, channles, 224, 224, requires_grad=True)
out_simple = simple_model(x)

# Export the model
torch.onnx.export(simple_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  simple_model_name,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=opset_version,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
)
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}})

onnx_simple_model = onnx.load(simple_model_name)
onnx.checker.check_model(onnx_simple_model)
# onnx.helper.printable_graph(onnx_simple_model.graph)

ort_session_simple = onnxruntime.InferenceSession(simple_model_name)

# compute ONNX Runtime output prediction
ort_inputs_simple = {ort_session_simple.get_inputs()[0].name: to_numpy(x)}
ort_outs_simple = ort_session_simple.run(None, ort_inputs_simple)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(out_simple), ort_outs_simple[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")



########### multiple inputs model ##########

multi_model = MultiInputConv()
multi_model.eval()
x = torch.randn(batch_size, channles, 224, 224, requires_grad=True)
y = torch.randn(batch_size, channles, 224, 224, requires_grad=True)
out_multi = multi_model(x, y)

torch.onnx.export(multi_model,
                  (x, y),
                  multi_model_name,
                  export_params=True,
                  opset_version=opset_version,
                  do_constant_folding=True,
                  input_names = ['input_x', 'input_y'],
                  output_names = ['output'],
)
                #   dynamic_axes={'input_x' : {0 : 'batch_size'},
                #                 'input_y' : {0 : 'batch_size'},
                #                 'output' : {0 : 'batch_size'}})

onnx_multi_model = onnx.load(multi_model_name)
onnx.checker.check_model(onnx_multi_model)
# onnx.helper.printable_graph(onnx_multi_model.graph)

ort_session_multi = onnxruntime.InferenceSession(multi_model_name)

ort_inputs_multi = {ort_session_multi.get_inputs()[0].name: to_numpy(x),
                    ort_session_multi.get_inputs()[1].name: to_numpy(y)}
ort_outs_multi = ort_session_multi.run(None, ort_inputs_multi)

np.testing.assert_allclose(to_numpy(out_multi), ort_outs_multi[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
