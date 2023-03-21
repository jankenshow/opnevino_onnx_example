# 概要

- pytorchのモデルをonnxに変換  
- OpenVINOからonnxのファイルを読み込んで、C++(Inference API)で推論を行う  
- 本レポジトリでは`dynamic shpaes`にはまだ非対応  

# モデルについて

- 2つの画像を入力として受け取って、1つの画像を出力する。  
- 画像のサイズは`3 x 224 x 224`(入出力すべて)  
- モデルの構造は `cat -> 1x1 conv -> ReLU`(詳細は[こちら](./convert2onnx.py#L29-L40))  


# 実行環境

## pytorchのモデルをonnxに変換

お好きなpython環境でもできるはず。  

- python            : 3.8.12  
- torch             : 1.13.1  
- onnx              : 1.13.1  
- onnxruntime       : 1.14.1  

onnx のインストールは下記コマンドで実行  

```
$ pip install onnx onnxruntime
```

## onnxのファイルを読み込んで、C++推論を行う

docker container 内でビルド・実行する。  
利用するイメージは[こちら](https://hub.docker.com/r/openvino/ubuntu20_dev)  



# 実行

## pytorchのモデルをonnxに変換

```
$ python convert2onnx.py
```

`simple_model.onnx`と`multi_model.onnx`の2つが、`data`ディレクトリに出力される。  
下記の OpenVINOでの推論には、`multi_model.onnx`を利用する。  


## onnxファイルを用いた、OpenVINOでの推論 (C++)

推論には`./data/[image1.png | image2.jpg]`を利用する。

### 環境作成

```
$ docker pull openvino/ubuntu20_dev
$ ./run_docker.sh
> /opt/intel/openvino_2022.3.0.9038/setupvars.sh
> cd /workspace
```

### ビルド

```
$ ./build.sh
```

### 実行

```
$ cd build
$ ./openvino_inference
```


# その他

- torchからonnxへの変換に関する[docs](https://pytorch.org/docs/stable/onnx.html)  
- OpenVINOでdynamic shapeのonnxモデルを推論する場合は、`InferenceEngine`ではなく`ov::Core:compile_model API`を利用する。  
- OpenVINOをローカル環境にインストールする際は[こちら](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html)  
- onnxrunime for c++のインストール方法は[こちら](https://onnxruntime.ai)  
