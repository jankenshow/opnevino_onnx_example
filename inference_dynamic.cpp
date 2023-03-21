#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/core.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <string>

void preprocess(const cv::Mat& input_image, ov::runtime::Tensor& input_tensor) {
  // 入力画像をリサイズ
  cv::Mat resized_image;
  cv::Size tensor_size(input_tensor.get_shape()[3],
                       input_tensor.get_shape()[2]);
  cv::resize(input_image, resized_image, tensor_size);

  // 画像をFP32に変換し、1/255スケーリングを適用
  cv::Mat fp32_image;
  resized_image.convertTo(fp32_image, CV_32F, 1.0 / 255);

  // cv::MatからTensorへのデータ転送
  float* input_data = input_tensor.data<float*>();
  for (size_t c = 0; c < 3; ++c) {
    for (size_t h = 0; h < tensor_size.height; ++h) {
      for (size_t w = 0; w < tensor_size.width; ++w) {
        input_data[c * tensor_size.width * tensor_size.height +
                   h * tensor_size.width + w] =
            fp32_image.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
}

int main() {
  // 入力画像の読み込み
  cv::Mat input_image1 = cv::imread("image1.jpg", cv::IMREAD_COLOR);
  cv::Mat input_image2 = cv::imread("image2.jpg", cv::IMREAD_COLOR);

  // OpenVINO推論エンジンの初期化
  ov::runtime::Core ie;
  const std::string model_path = "path/to/your/onnx/model.onnx";
  ov::runtime::ExecutableNetwork executable_network =
      ie.compile_model(model_path, "CPU");
  ov::runtime::InferRequest infer_request =
      executable_network.create_infer_request();

  // 入力画像の前処理とTensorへのデータ転送
  const std::string input_name1 = "input_name1";
  const std::string input_name2 = "input_name2";
  ov::runtime::Tensor input_tensor1 = infer_request.get_tensor(input_name1);
  ov::runtime::Tensor input_tensor2 = infer_request.get_tensor(input_name2);
  preprocess(input_image1, input_tensor1);
  preprocess(input_image2, input_tensor2);
  infer_request.set_tensor(input_name1, input_tensor1);
  infer_request.set_tensor(input_name2, input_tensor2);

  // 推論の実行
  infer_request.infer();

  // 結果の取得
  const std::string output_name = "output_name";
  ov::runtime::Tensor output_tensor = infer_request.get_tensor(output_name);

  // 出力Tensorからデータを取得
  const float* output_data = output_tensor.data<float*>();

  // 出力データをcv::Matに変換
  size_t out_height = output_tensor.get_shape()[2];
  size_t num_channels = output_tensor.get_shape()[1];
  cv::Mat output_image(out_height, out_width, CV_32FC3);

  for (size_t c = 0; c < num_channels; ++c) {
    for (size_t h = 0; h < out_height; ++h) {
      for (size_t w = 0; w < out_width; ++w) {
        output_image.at<cv::Vec3f>(h, w)[c] =
            output_data[c * out_width * out_height + h * out_width + w];
      }
    }
  }

  // 結果の画像を保存
  cv::Mat output_image_8UC3;
  output_image.convertTo(output_image_8UC3, CV_8UC3, 255);
  cv::imwrite("output_image.jpg", output_image_8UC3);

  return 0;
}
