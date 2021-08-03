# OCT 이미지를 이용한 황반질환 분류 프로젝트


![syed-hussaini-3nQ4pIOW2g4-unsplash](https://user-images.githubusercontent.com/80465347/127896461-dc5f95bb-1912-4769-a1d9-c2f96715b5d4.jpg)

'황반'이라는 이름은 생소하지만, 망막에서도 시신경과 시세포가 밀집해있고 초점이 맺히는 곳이 바로 황반이기 때문에 그만큼 중요하고 또 예민한 부위입니다. 우리의 시력을 담당한다고도 할 수 있습니다. 그런 만큼 황반에 질환이 생기게 되면 시력이 손상되고 심한 경우 실명으로 이어질 수도 있다고 합니다. 
문제는 이러한 황반 질환의 경우 조기 진단과 치료가 필요함에도 불구하고 정작 증상 초기에는 뚜렷한 증상이 없거나, 흔히는 노안 증상과 착각하여 진료가 늦어진다는 점입니다.
이에 이번 프로젝트를 통해 빠른 진단에 도움이 되도록 OCT 이미지에 기반하여 황반 질환을 분류하는 CNN 모델을 만들어보고자 하였습니다.  
(참고. 프로젝트의 세부적인 내용은 하단의 "최종 발표자료" 링크에서 확인하실 수 있습니다.)

## 1. Project Summary 
### 1.1 목적 
황반변성 질환(CNV, DME, Drusen) 분류하기 

### 1.2 결과
이번 프로젝트에서 최종적으로 가장 성능이 좋았던 케이스는 validation accuracy 0.9802를 기록하였습니다. 
해당 케이스는 다음과 같습니다.
- 이미지 처리 : 이미지 중심 이동, CLAHE 적용  
- CNN : InceptionResNetV2, Imagenet 가중치를 이용한 전이학습
샘플링을 통해 이미지 데이터셋의 사이즈가 줄어들었다는 것을 감안하면, 충분히 높은 성능으로 볼 수 있을 것 같습니다.  
(참고. Kaggle에 공개되어있는 모델 중 가장 성능이 좋은 모델이 validation accuracy 0.99를 기록하였습니다.)

### 1.3 데이터

- Retinal OCT Image Dataset / [Kaggle](https://www.kaggle.com/paultimothymooney/kermany2018) - University of California San Diego

### 1.4 진행순서 

- 데이터셋 탐색
- 샘플링
- 베이스라인 선정
- 이미지 처리 기법 적용  
- CNN 모델 적용  

### 1.5 주요 이미지 처리 기법 및 CNN 모델 

- 이미지 처리 
  - CLAHE / HE
  - Difference of Gaussians
  - HSV mask
  - Image Contour
- CNN 모델
  - LeNet
  - VGG16, VGG19
  - MobileNet
  - ResNet50
  - GoogleNet, InceptionV3
  - InceptionResNetV2 

## 2. 세부내용
### 2.1 데이터셋 탐색
- 데이터셋의 Class는 총 4개로, CNV, DME, Drusen 그리고 Normal Class입니다.
- Train Set 83,484장, Test Set 1,000장이 있으며, 각 파일 제목은 "Class-환자ID-사진번호"의 형식으로 구성되어 있습니다.(예시: CNV-53018-1.jpeg)
- 그런데 제공된 데이터셋에서 데이터 오염의 가능성이 있는 문제점들을 세 가지 발견하였습니다. 
  1) 한 환자가 두 가지 증상을 동시에 보이는 경우가 존재하였습니다.
  2) 동일 환자가 Train Set에도, Test Set에도 포함되어 있었습니다.
  3) 환자당 사진의 수가 불균형하여, 적게는 1장, 많게는 한 환자당 800장까지도 반영되어 있었습니다.
- 이에 샘플링을 진행하기로 결정하였습니다.
 
### 2.2 샘플링
- 두 개 이상의 증상을 가진 환자들을 제외하기 위해, 단일 증상 환자의 데이터만 필터링하였습니다.
- 기존의 Test set을 사용하지 않고, 환자 ID를 기준으로 Train Set과 Test Set을 다시 구분하였습니다. 다시 말해, 샘플링을 거친 후 Train Set에 포함된 환자는 Test Set에 속하지 않습니다.
- 환자 ID 1개 당 이미지 1장으로 랜덤 샘플링을 진행하였습니다.
- 최종적으로 7,099장의 이미지를 사용하게 되었습니다.

### 2.3 베이스라인 선정
- 베이스라인 선정을 위해 적용했던 CNN 모델들은 LeNet, VGG16, VGG19, 그리고 MobileNet입니다.
- 이 중 LeNet, VGG16, VGG19는 학습이 잘 진행되지 않았고, MobileNet은 validation accuracy 0.8801을 기록하였습니다. 
- 이에 0.8801을 기준으로 이미지 처리와 다른 CNN 모델 적용을 통해 성능을 높여가기로 하였습니다. 
 
### 2.4 이미지 처리 기법 적용

- 이미지 처리는 기본적으로 CLAHE, DoG, Contour, HSV 조절 의 방법을 사용하였으며, 추가적으로 DoG를 변형하여 원본 이미지에서 가우시안 필터를 적용한 이미지를 빼는 방법을 사용하였습니다.(임의로 Subtraction이라는 명칭을 사용하였습니다.)
- 위 방법 중 두 개 이상 조합하거나, 이미지 이진화 등 다른 기법과 함께 조합하기도 했습니다.
- 또한 이미지가 치우쳐있는 경우가 많아 중앙으로 이동시키는 것도 진행해보았습니다.
- 결과적으로 원본 이미지를 중앙으로 이동시켰을 때, 그리고 CLAHE를 적용했을 때 모델의 성능이 가장 좋았습니다.  
 
### 2.4 CNN 모델 적용 
- 랜덤 샘플링을 통해 이미지 개수를 대폭 감소하였기 때문에, Transfer Learning을 사용하였으며,
Fine Tuned Convolution Network를 사용해 CNN의 Fully Connected Layer를 변경하여 사용하였습니다.
- CLAHE, Transfer Learning, 이미지 중앙화를 고정옵션으로 두고, Best Accuracy를 찾기 위해 다양한 모델을 적용해 보았습니다.
- 여러 모델 적용 중 Res Net, Inception 계열의 모델들이 높은 Accuracy를 보여주었습니다.
- Drusen 분류에는 Inception v3 모델 성능이 가장 좋았습니다.
- Inception v4, Residual Network를 결합한 Inception Res Net v2 모델은 전체적인 성능(4 class 분류)에서 가장 좋았습니다.


## 3. File List 

-  ImgFilter.py : 이미지 처리를 위한 모듈 
-  img_preprocess.py : 사용한 이미지 처리 기법들 
-  img_preprocess_fail.py : 실패한 이미지 처리 기법들
-  CleanModelV2.py : 딥러닝 처리를 위한 py

## 3. Contributors

* [정현](https://github.com/JeonghyunKo) - 데이터 샘플링 / 이미지 처리 
* [동주](https://github.com/lee-edgar) - CNN 모델 적용 

## 4. What We Learned, and more... 

- Kaggle 공개 코드들에서는 데이터셋 탐색을 진행한 경우를 찾아보지 못했지만, 충분히 문제가 될 수 있는 케이스들이 다수 존재하였습니다. 데이터셋 탐색부터 꼼꼼하게 진행하는 것이 중요하다는 것을 다시금 배웠습니다.
- 학습 데이터의 양이 충분하지 못할 경우 Transfer Learning이 필요하며, Transfer Learning은 모델 튜닝과 Fine Tuned Convolution Network방식을 사용하여 Deep한 Layer를 구성하여 더 좋은 성능을 이끌어 낼 수 있다는걸 배웠습니다. 앞으로 Transfer Learning에 대해서 더 연구하고자 합니다. 

## 5. Acknowledgments 

* [Flycode77](https://github.com/FLY-CODE77)  
* [PinkWink](https://github.com/PinkWink) 

## 프레젠테이션 

* [최종 발표자료](https://drive.google.com/file/d/14zlx6sMUkVf-FzUKgFtDZxHWyBqisQj4/view?usp=sharing)
* [이미지 출처](https://unsplash.com/photos/3nQ4pIOW2g4?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
