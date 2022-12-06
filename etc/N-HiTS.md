# 개요
Time Series 데이터에 대한 forecasting 작업을 위한 최신의 모델로 N-HiTS라는 것이 제안되었다.
본 모델은 N-Beats가 가지고 있던 계산비용이 크다는 이슈를 최적화 하는 동시에 prediction 성능을 향상 시킬 수 있다고 말하고 있다.

참조
https://www.datasciencewithmarco.com/blog/the-easiest-way-to-forecast-time-series-using-n-beats
https://arxiv.org/abs/1905.10437

# N-Beats
![1_dfmx9FMpWHJR61RKpfsk0A](https://user-images.githubusercontent.com/37214630/205844137-4ac91e7a-1629-4af4-bdba-e370daf562fc.png)

상기 이미지에서 확인할 수 있듯이, N-Beats 모델의 경우, 입력 데이터에 대해서 각각 블록을 거쳐가며, 각각 forecast output과 backcast output을 내놓으면서,
block의 backcast output의 경우 입력과 잔차만 반환함으로써, 현재 블록에서 잡아내지 못한 특징이 포함된 데이터만 다음 블록으로 넘어간다고 설명하고 있다.

또한 논문에는 두가지 유형의 블록을 제안하며, Interpretable을 위한 성능을 향상시킨 블록을 제안하고 있는데, 해당 블록은
각각 블록 내부에서 trend 한 output과 seasonal 한 output을 개별적으로 계산하여, 추후 model output을 보일때, 각각 trend와 seasonal한 특징을 볼 수 있는 장점이 있다.
하지만 실제 시계열 데이터의 경우 trend와 seasonal에 대한 정보를 얻을 수 없고, 주기성과 같은 특징을 찾을 수 없는 경우가 많아 바로 적용하기는 힘든점이 있다.

# N-HiTS
![캡처](https://user-images.githubusercontent.com/37214630/205845042-58796ceb-a526-4d85-b30f-e259ddf7f9c2.PNG)
(https://arxiv.org/abs/2201.12886)

N-Beats와 거의 유사하나, 각각 블록에는 MaxPooling layer를 추가하여, 각각 블록에 데이터에 대한 sampling rate를 추가하는 효과를 얻을 수 있다고 알려준다.
이를 통해서 모든 블록에서 모든 time 에 대해서 계산을 수행하는 N-Beats보다 몇배나 적은 수의 계산을 수행할 수 있다는 장점이 있다. (이를 위해서 tensorflow에 reshape연산이 추가되는건 내가 잘못하는 거겠지?)
나머지 대부분의 구조는 N-Beats의 Generic한 Block의 경우와 동일하다.

임의의 데이터에 대해서, 동일한 규모의 파라미터 수로 구성된 단순하 dense와 간단하게 가능한 간단하게 구현한 n-hits의 결과를 보면 아래와 같다.
![Figure_1](https://user-images.githubusercontent.com/37214630/205846841-1227e2d0-ecd1-4a18-9cc5-a9b8cce4f1ff.png)

그래프상 결과만 놓고 봤을때, 특정 시점에 발생하는 overshoot에 대한 예측이 N-HiTS의 forecasting 능력이 단순한 dense에 비해서는 압도적으로 좋은 것을 볼 수 있다.
'''
N-Hits
inputs = tf.keras.layers.Input((H * N,))

forecast_list = []
backcast_list = []
forecast, backcast = block(inputs, 32, H, N)
forecast_list.append(forecast)
backcast_list.append(backcast)
for i in range(4):
    forecast, backcast = block(backcast_list[-1], 32 // (2**(i+1)), H, N)
    forecast_list.append(forecast)
    backcast_list.append(backcast)
output = tf.keras.layers.Add()(forecast_list)
model = tf.keras.Model(inputs, output)
'''

'''
dense
inputs = tf.keras.layers.Input((H * N,))

x = inputs

for i in range(3):
    x = tf.keras.layers.Dense(64 * (2 ** (i+1)), activation='relu', kernel_initializer="glorot_uniform")(x)
    x = tf.keras.layers.Dense(64 * (2 ** (i+1)), activation='relu', kernel_initializer="glorot_uniform")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

output = tf.keras.layers.Dense(H)(x)

model = tf.keras.Model(inputs, output)
'''
