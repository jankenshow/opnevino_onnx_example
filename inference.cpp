#include <inference_engine.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

void preprocess(const cv::Mat& input_image,
                InferenceEngine::Blob::Ptr& input_blob) {
  // input_blobのサイズを取得
  const size_t width = input_blob->getTensorDesc().getDims()[3];
  const size_t height = input_blob->getTensorDesc().getDims()[2];

  // 入力画像をリサイズ
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, cv::Size(width, height));

  // 画像をFP32に変換し、1/255スケーリングを適用
  cv::Mat fp32_image;
  resized_image.convertTo(fp32_image, CV_32F, 1.0 / 255);

  // cv::MatからBlobへのデータ転送
  float* input_data = input_blob->buffer().as<float*>();
  for (size_t c = 0; c < 3; ++c) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        input_data[c * width * height + h * width + w] =
            fp32_image.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
}

int main() {
  try {
    // 1. ONNXファイルをロードする
    InferenceEngine::Core ie;
    const std::string model_path = "../data/multi_model.onnx";
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(model_path);
    InferenceEngine::ExecutableNetwork executable_network =
        ie.LoadNetwork(network, "CPU");
    InferenceEngine::InferRequest infer_request =
        executable_network.CreateInferRequest();

    // 2. OpenCVのMat(画像)を入力とする
    cv::Mat input_image1 = cv::imread("../data/image1.png");
    cv::Mat input_image2 = cv::imread("../data/image2.jpg");
    if (input_image1.empty()) {
      std::cerr << "Error: Could not open image1.png file." << std::endl;
      return 1;
    }
    if (input_image2.empty()) {
      std::cerr << "Error: Could not open image2.png file." << std::endl;
      return 1;
    }

    // 3. ONNXモデルは、二つの画像を受け取って同じ大きさの画像を返す
    InferenceEngine::InputInfo::Ptr input_info1 =
        network.getInputsInfo().begin()->second;
    InferenceEngine::InputInfo::Ptr input_info2 =
        (++network.getInputsInfo().begin())->second;
    std::string input_name1 = network.getInputsInfo().begin()->first;
    std::string input_name2 = (++network.getInputsInfo().begin())->first;

    // 入力画像の前処理とBlobへのデータ転送
    // input_info1->setPrecision(InferenceEngine::Precision::FP32);
    // input_info2->setPrecision(InferenceEngine::Precision::FP32);
    // input_info1->setLayout(InferenceEngine::Layout::NHWC);
    // input_info2->setLayout(InferenceEngine::Layout::NHWC);

    // infer_request.SetBlob(input_name1,
    //                       InferenceEngine::make_shared_blob<float>(
    //                           input_info1->getTensorDesc(), image1.data));
    // infer_request.SetBlob(input_name2,
    //                       InferenceEngine::make_shared_blob<float>(
    //                           input_info2->getTensorDesc(), image2.data));

    InferenceEngine::Blob::Ptr input_blob1 = infer_request.GetBlob(input_name1);
    InferenceEngine::Blob::Ptr input_blob2 = infer_request.GetBlob(input_name2);
    preprocess(input_image1, input_blob1);
    preprocess(input_image2, input_blob2);

    // 推論の実行
    infer_request.Infer();

    // 出力
    // const float* output_data = output_blob->buffer().as<const float*>();
    // size_t output_data_size = output_blob->size();
    InferenceEngine::Blob::Ptr output_blob =
        infer_request.GetBlob(network.getOutputsInfo().begin()->first);

    // 出力Blobからデータを取得
    const float* output_data = output_blob->buffer().as<const float*>();

    // 出力データをcv::Matに変換
    // cv::Mat result(input_image1.rows, input_image1.cols, CV_8UC3,
    //                (void*)output_data);
    size_t out_width = output_blob->getTensorDesc().getDims()[3];
    size_t out_height = output_blob->getTensorDesc().getDims()[2];
    size_t num_channels = output_blob->getTensorDesc().getDims()[1];

    cv::Mat output_image(out_height, out_width, CV_32FC3);
    for (size_t c = 0; c < num_channels; ++c) {
      for (size_t h = 0; h < out_height; ++h) {
        for (size_t w = 0; w < out_width; ++w) {
          output_image.at<cv::Vec3f>(h, w)[c] =
              output_data[c * out_width * out_height + h * out_width + w];
        }
      }
    }

    // 結果画像の後処理
    cv::Mat result;
    output_image.convertTo(result, CV_8UC3, 255);

    // 結果をファイルに保存
    cv::imwrite("../data/result.png", result);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown/internal exception happened." << std::endl;
    return 1;
  }

  std::cout << "Inference finished successfully!" << std::endl;
  return 0;
}
