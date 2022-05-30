# 개요
기존 one-class network 방식의 svdd나 oc-svm 과 같은 모델은 시계열 데이터를 하나의 벡터로 변환하여, 
해당 벡터에 대해서만 이상감지를 수행함으로 인해서 시계열적인 정보에 대한 손실이 많이 발생한다.
이를 개선하기 위해서 RNN 기반의 neural network를 활용하는 모델이 많이 제안되고 있으며, 이 논문도 그 중 하나이다.
이 논문은 시계열 특징을 데이터 대해서, dilated-rnn 구조를 활용해서, layer의 depth에 따라서 dilate를 줌으로 인해서 
보다 낮은 해상도에서 보다 장기간의 데이터의 특징을 얻어내고, 높은 해성도에서 보다 짧은 기간의 데이터의 특징을 얻어내도록 모델이 학습 되도록 구성
# THOC
위의 dilated-rnn을 통해서 각각 resolution에 대해서 특징을 얻어낸 hidden feature들에 대해서 각 레이어에 학습 중인 cluster center 와 cosine 유사도를 계산하여, 가중치로 사용하고,
cluster center의 차원와 동일하게 해당 step의 output에 matmul을 수행하여, 다음 layer로 전달한다. 마지막 layer에 대해서는 다음 layer의 dilated-rnn의 output이 없기 때문에,
따로 concat을 수행하지 않고, f_hat 으로 마지막 레이어의 클러스터와 코사인 거리를 계산하고, 그 거리에 각각 코사인 유사도의 softmax를 곱해서, 코사인 유사도가 높은 것의 거리에
더 높은 가중치를 두게 loss가 형성된다.
## architecture
![Screenshot 2022-05-30 at 14 41 47](https://user-images.githubusercontent.com/37214630/170924766-d53ee89f-dc51-4442-8619-cfaee3072f70.jpg)
## feature vector
![Screenshot 2022-05-30 at 14 42 10](https://user-images.githubusercontent.com/37214630/170924797-e75c52c1-ee25-43ab-bf08-7d1c98fad32b.jpg)
![Screenshot 2022-05-30 at 14 42 24](https://user-images.githubusercontent.com/37214630/170924802-4ecfb542-5307-49cc-bb45-ccd0799acf96.jpg)
![Screenshot 2022-05-30 at 14 42 29](https://user-images.githubusercontent.com/37214630/170924805-a128ddb3-156a-41b2-9d9a-89e21ac6ae09.jpg)
![Screenshot 2022-05-30 at 14 42 34](https://user-images.githubusercontent.com/37214630/170924808-54d8dc72-d179-476d-ad1b-a73b886428dc.jpg)
![Screenshot 2022-05-30 at 14 42 42](https://user-images.githubusercontent.com/37214630/170924811-4bb4c1cf-f551-44db-8dfd-088ebc5adda1.jpg)
## loss
![Screenshot 2022-05-30 at 14 42 47](https://user-images.githubusercontent.com/37214630/170924819-65bb0d28-129f-458f-902e-0b79f026858f.jpg)
![Screenshot 2022-05-30 at 14 42 53](https://user-images.githubusercontent.com/37214630/170924821-d2700297-d79f-42ed-ac52-2a2479195cf4.jpg)
![Screenshot 2022-05-30 at 14 42 58](https://user-images.githubusercontent.com/37214630/170924823-3abe1a54-f6a5-4805-8ded-6e994fbaa1a0.jpg)

# 조금더 상세한 설명
위의 architecture에서 각각 feature vector를 얻어내고, layer의 hidden vector들을 다시한번 feature fector로 변환하는 작업을 진행해서 마지막으로 loss를 계산하게 된다.
상세한 모델 파라미터나 클러스터 갯수등은 논문에 포함되어 있지 않다 (아마 이래서 github 구현코드가 따로 없는듯, 실험 제현이 불가능함)
하지만 각각 rnn-architecture의 가중치와 클러스터 포인트들을 한번에 학습해서, cluster로 embedding 하는 모델과 cluster point를 자동으로 한번의 학습으로 찾을 수 있다는 장점을 가진 모델이다.
상세한 구현이나 계산은 생략된 내용이 너무 많아 진행이 힘들지만 concept정도는 참고가 가능할 것 같다.
