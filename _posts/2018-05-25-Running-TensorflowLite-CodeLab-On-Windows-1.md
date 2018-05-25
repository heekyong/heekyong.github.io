---
layout: post
title:  "Running-TensorflowLite-CodeLab-On-Windows-1"
date:   2018-05-25 17:22:47 +0900
categories: jekyll update
---

## 모바일 환경에서 딥러닝 모델 사용하기

  텐서플로우를 사용하여 이미 트레인한 딥러닝 모델을 모바일 환경에서도 사용할 수 있도록,
즉 이미지 인식 등의 딥러닝 모델을 앱 개발에서 쉽게 사용할 수 있도록
텐서플로우에서는 텐서플로우 모바일 TFMobile과 텐서플로우 라이트 TFLite를 공개했습니다.
텐서플로우에서 제공하는 TF Lite와 TF Mobile의 사용법을 정리한 CodeLab이 2개 있습니다.
[Tensorflow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/), [Tensorflow For Poets2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0).
리눅스 환경을 기준으로 만들어진 이 두가지 코드랩을 윈도우 환경에서 따라해보고 싶은 분들을 위해 작성해보았습니다.

* * *


## TF Mobile과 TF Lite의 차이점?
1. 텐서플로우 모바일은 라이트보다 훨씬 더 먼저 나온 버전이고
텐서플로우 라이트는 조금 더 경량 모델로, 2018년에 새로 공개된 것으로 현재 공개된 것은 개발자 프리뷰버전입니다.

2. 텐서플로우 모바일과 라이트 공식 홈페이지에 따르면,
TF Lite는 Mobile에 비해 바이너리가 더 작고 디펜던시도 적으며 퍼포먼스는 더 좋을 것이라고 합니다.
다만 텐서플로우 라이트는 현재 InceptionV3와 MobileNet 아키텍처만을 지원하고 있습니다.

3. 파일 형식의 차이: 텐서플로우 모바일은 PC에서 텐서플로우를 통해 트레인한 모델의 graph.pb파일과 레이블 텍스트파일을 그대로 앱에서 불러와서 사용합니다. 텐서플로우 라이트는 트레인된 그래프 파일(pb형식)과 레이블을 TOCO(Tensorflow lite용 컨버터)를 이용하여
tflite라는 형식의 파일로 변환한 후에 그 lite파일을 모바일 앱에서 사용합니다.
##### 그런데 2018년 5월 현재 기준 TOCO의 윈도우 지원이 돼지 않는다고 텐서플로우팀에서 밝혔습니다. [참고](https://github.com/tensorflow/tensorflow/issues/16374)
따라서 윈도우 환경에서 개발해야하는 경우, 코드랩2의 변환 과정은 생략하고 grpah.pb파일 그대로 TFMobile 앱으로 사용하거나
그 파일 변환 부분만 리눅스 환경에서 따로 진행해야합니다.

* * *

: 코드랩에 들어가기 전에.

## Transfer Learning 과 Retrain


CNN 모델의 일종인 모바일넷, 인셉션 등의 딥러닝 모델을 이미지 인식을 위해 처음부터 끝까지 트레이닝 시키기 위해서는 일반 컴퓨터가 감당하기에는 아주 많은 시간과 자원이 듭니다.
따라서 코드 랩에서는 누구나 쉽게 따라해 볼 수 있도록 하기 위해, 이미 ImageNet 데이터셋으로 fully trained된 모델의 최상위 레이어만 사용자가 원하는 데이터셋으로 retrain하는 단계를 거치고 있습니다.

pre-trained된 전체 네트워크의 하위레이어는 class-object specific하지 않을 것이라는 가정 하에,
이미지넷의 1000여개 사물, 동물 클래스에 대한 이미지를 가지고 트레인된 모델을 사용해서
최상위 레이어만 re-train하는 것을 Transfer Learning이라고 합니다.(트랜스퍼러닝에는 최상위 레이어만 리트레인하는 것 외에 상위 레이어를 fine-tuning하는 등 여러가지 방법이 있지만 현재 코드랩에서 다루고 있는 부분까지만 설명하겠습니다.)

>즉 트랜스퍼 러닝은 성능이 입증된 fully pre-trained CNN을 가져다가 feature를 추출하는 데에 사용하고,
이를 바탕으로 우리가 원하는 분류(커스텀 데이터셋)를 수행하도록 만드는 것이라고 볼 수 있습니다.

이러한 트랜스퍼 러닝의 장점은
1. __적은 데이터셋__(클래스당 1000개 미만)으로도 오버피팅을 피하며 효율적 트레이닝 가능하다는 점. 라벨링 된 양질의 데이터를 충분히 모으는 것이 쉽지 않은데, 수백개 정도의 데이터로도 충분히 높은 퍼포먼스를 보입니다.
2. __속도 및 비용__ 현격하게 절약. 이미지넷과 같은 딥러닝용 대규모 데이터로 CNN전체 모델을 학습시킬 때 컴퓨터 사양에 따라 며칠, 몇 달이 걸리지 모르는 것와 비교하여 최상위 레이어만 리트레인할 경우 수 분내에 리트레이닝 과정이 끝나는 것을 코드랩을 통해 확인할 수 있을 것입니다.

밑에서 진행하게 될 코드랩은 이미지넷 데이터셋으로 1000개의 사물, 동물 클래스에 대해 트레이닝 된 모델을 가져다가
daisy, dandelion, rose 등 다섯가지 종류의 꽃 이미지를 가지고 retrain하여
그 다섯가지 꽃 종류를 분류하는 모델을 만들어 그것을 모바일에서 사용하는 과정입니다.

* * *

코드랩을 시작하기 전에 컴퓨터에 python, git 및 tensorflow1.7 버전은 이미 설치된 상태여야합니다.

### 코드랩 1. 준비 단계

윈도우 커맨드창(명령프롬프트)를 열고 원하는 디렉토리에 tensorlfow-for-poets-2 파일들을 내려받은 후 그 폴더로 들어갑니다.

~~~
  git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
  cd tensorflow-for-poets-2
~~~

저는 다운로드 폴더에서 진행하였습니다.
이렇게 다운로드 폴더에 코드랩 파일들이 다운받아진 것을 확인해볼 수 있습니다.

![tfmobile1](../../../../../images/tfmobile1.png)

![tfmobile2](../../../../../images/tfmobile2.png)

그 다음이 리트레인을 하기 위한 이미지셋으로 쓸 꽃 사진들을 다운받습니다. 리눅스 환경에서는 curl을 사용하여 커맨드창에서 바로 받아서 원하는 위치에 압축까지 해제할 수 있지만, 윈도우에서는 인터넷 주소창에 아래 주소를 복사해넣으면 파일을 다운받는 방법을 씁시다.
~~~
http://download.tensorflow.org/example_images/flower_photos.tgz
~~~
flower_phohtos.tgz 파일이 다운 받아지면, 현재 작업하고 있는 폴더인 tensorflow-for-poets-2 내부의 tf_files 폴더에 압축을 풀어주세요.
비어있던 tf_files폴더에 flower_photos라는 폴더 아래 각 꽃 종류별 이미지 데이터셋이 저장된 것을 확인해주세요.
![tfmobile3](../../../../../images/tfmobile3.png)
~~~
cd tf_files/flower_photos
dir
~~~
위와 같이 커맨드창에서 바로 확인해볼 수도 있습니다.
![tfmobile4](../../../../../images/tfmobile4.png)
