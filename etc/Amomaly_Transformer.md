# 개요
Time Series 데이터에 대한 anomaly detection을 수행하는데 있어서, Anomaly Transformer 기법을 적용하면, 단순한 Reconstruction이 아닌, 
association discrepancy 라는 데이터와 딥러닝 모델을 통해서, 데이터가 가지고 있는 이상치를 수치화 할 수 있고, 이 값과, reconstruction error를 활용해서 anomaly score를 계산할 수 있다.
이때 association discrepancy는 본 논문에서 제안한 anomaly transformer 블록을 통해서 계산할 수 있다.

# Anomaly Transformer
![Screenshot 2022-05-26 at 15 28 02](https://user-images.githubusercontent.com/37214630/170430477-28347650-fc91-4b38-a016-9bd9ae15f8a8.jpg)

위 그림을 보면 입력 벡터 X ~ (timw_window_length, feature_number) 의 입력을 받아서 output z 를 내놓는 block을 여러개 쌓고 마지막 layer의 output을 reconstruction으로 계산하는 모델로
입력 벡터 X에 (feature_number , 1) 차원의 가중치를 곱해서 sigma ~ (timw_window_length, 1) 차원의 벡터를 얻은 뒤, 아래의 Rescale 식을 통해서 N, N 의 데이터의 확률 분포 정보를 얻어낸다. (Prior-Association)
그리고 Self-attention 기법을 통해서 Query와 Key를 얻고 그를 곱해서 N, N 차원의 hidden feature를 얻어내고(Series-Association),
이 텐서에 Softmax를 마지막 차원에 대해서 적용하여 중요한 시점의 위치만 강조하게 만든다. 이제 이를 Prior-Association과 Series-Association의 분포가 동일하게 되게끔
가중치 들을 학습할 수 있게 Association Discrepancy라는 이름의 목적함수로 설정하는데, Prior-Association과 Series-Association의 KLD의 값을 나타낸다.
![Screenshot 2022-05-26 at 15 36 57](https://user-images.githubusercontent.com/37214630/170431893-580a1afa-cde1-446a-ad90-7f8f04311b7b.jpg)
![Screenshot 2022-05-26 at 15 37 06](https://user-images.githubusercontent.com/37214630/170431896-678f87f7-5b78-4d34-a532-607370b5d77c.jpg)
실제 복원할 때는 Series-Association만을 활용해서, Series-Association (time_window_length, time_window_length) 과 또다른 hidden feature V 두 텐서를 matmul하여 
(time_window_length, hidden_feature_num) 차원의 tensor를 block의 output으로 내놓게 된다. 

자 만약 이상데이터가 들어오게 된다면, 이 Prior-Association과 Series-Association이 동일하게 구해지지 않을 것이며, 추가적으로 복원한 데이터 역시 입력 데이터와 차이가 클 것이기 때문에,
본 모델을 통해서 보다 정밀하게 이상을 감지하게 될 것으로 가정하고 있다.

또한 block을 여러개 겹쳐서 multi-head로 수행하는 경우, 각각 head가 전체 범위의 일부분만 복원하고 concat하는 식으로 구성하게 된다. (즉 긴 벡터를 조각내서 각각 head에 넣고 복원하는 식)


Attention Block을 지난 뒤에는 Layer Norm 과 Feed Forward 를 통과한 뒤, 다시 Layer Norm을 보내는 식으로 Block이 구성되며 해당 Block이 반복되어서 하나의 Network를 구성한다.
이때 대부분의 Layer / Block의 Output이 동일하게 유지되어 skip-connection이 연결되어 있다.
