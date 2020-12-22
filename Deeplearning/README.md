# DeepLearning - Guide

※ Object Detection 딥러닝 모델로 faster-RCNN-resnet101 모델을 사용하여 학습할 경우 tflite 변환에 오류가 있을 수 있어(모델이 무겁기 때문에) SSD Mobilnet v1 모델을 사용하였다

## 1. Training Process
- 환경 setting
    1. tensorflow/modes 다운로드 후 research 폴더로 이동
    ```
    git clone http://github.com/tensorflow/models
    cd models/research
    ```

    2. 프로토콜 버퍼를 이용하기위한 설정(?)후 PYTHONPATH 환경 변수 설정
    ```
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```

    3. 설치 확인
    ```
    python object_detection/builders/model_builder_test.py
    ```

- model training
    1. 사용할 model , dataset 구축
        - Deeplearning Model : MSCOCO dataset에 대해 학습한 ssd_mobilnet_v1 사용
        - Dataset : Oxford-lllT Pets라는 데이터셋 이용(37종의 강아지, 고양이 사진 class별 200장씩 구성)
  
    2. dataset TFrecord 파일 만들기
        ```
        python object_detection/dataset_tools/create_pet_tf_record.py \
        --label_map_path=object_detection/data/pet_label_map.pbtxt \
        --data_dir=pwd \
        --output_dir=pwd
        ```

    3. train.py 코드 수정
        ```
        TRAIN_DIR = 'train_result' # path to save training result (저장할 경로)
        PIPELINE_CONFIG_PATH = 'pipeline_pet.config' # path to training configuration file (참고할 .config 파일)
        ```

    4. pipeline_pet.config 파일 수정 후 학습 진행\
        config 파일 : `models/research/object_detection/sameples/configs/ssd_mobilenet_v1_pets.config` 사용\
        learning_rate, fine_tune_checkpoint, num_step, input_path, label_map_path 수정\
        ```
        python train.py
        ```
        결과 result_training 폴더 생성 후 training된 ckpt 파일이 저장되어 있음

    5. 저장된 .ckpt모델 .pb파일로 저장\
        export_inference_graph.py 코드 수정\
        `PIPELINE_CONFIG_PATH = 'ssd_mobilenet_v1_pets.config'`
        `OUTPUT_DIR = 'export_pet'`
        `CKPT_PREFIX = 'result_training/model.ckpt-200000'`\
        코드 실행
        ```
        python export_inference_graph.py
        ```
        export_pet 폴더가 생성되고 안에 frozen_inference_graph.pb 파일과 ckpt파일, pipeline.config 파일이 생성됨

    6. 학습된 model test\
        object_detectoin_run.py 코드 수정\
        파일 경로 변수 파일 위치에 맞게 수정\
        ```
        python object_detection_run.py
        ```

## 2. .pb → .tflite 변환
참고사이트 : [https://gist.github.com/apivovarov/eff80275d0f72e4582c105921919b852](https://gist.github.com/apivovarov/eff80275d0f72e4582c105921919b852)

1. 학습이 완료 후 생성된 file들(.pb .ckpt pipeline.config)로 tflite.graph파일 생성

    ```
    python3 export_tflite_ssd_graph.py \
    --pipeline_config_path=export_pet/pipeline.config \
    --trained_checkpoint_prefix=export_pet/model.ckpt \
    --output_directory=/tmp/tflite_graph \
    --add_postprocessing_op=true
    ```

    - `--add_postprocessing_op = true`\
    output tensor : TFLite_Detection_PostProcess, TFLite_Detection_PostProcess:1, TFLite_Detection_PostProcess:2, TFLite_Detection_PostProcess:3
    - `--add_postprocessing_op = false`\
    output tensor : raw_outputs/class_predictions,raw_outputs/box_encodings

    위 코드 실행 → tflite_graph.pb, tflite_graph.pbtxt 파일 생성

2. .tflite 파일 생성
    ```
    tflite_convert \
    --graph_def_file=export_pet/tflite_graph.pb \
    --output_file=Tflite_model/model.tflite \
    --output_format=TFLITE \
    --input_arrays=normalized_input_image_tensor \
    --input_shapes=1,300,300,3 \
    --inference_type=FLOAT \
    --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
    --allow_custom_ops
    ```

    위 코드 실행 → model.tflite 파일이 생성된다.\
    `run_tflite.py` 파일로 sample 실행

3. meta data를 쌓는 작업 진행

    `pack_metadata.py` 실행\
    meta data가 씌여진 tflite파일이 만들어짐

## 3. android studio에서 학습모델 적용
- tflite예제에서 label.txt 파일, detect.tflite 파일을 학습시킨 모델의 tflite와 label.txt로 변경(ctrl + shift + r 을 누르면 전체 파일에서 찾기)

- DetectorActivity에서 `private static final boolean TF_OD_API_IS_QUANTIZED` 를 `false` 로 설정

- 실행