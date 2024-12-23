# 모듈
import tensorflow as tf
import matplotlib.pyplot as plt

# 손글씨 데이터셋 시각화

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 훈련셋 60000, 테스트셋 10000

print(x_train.shape, y_train.shape)  # (데이터셋 크기, 세로 픽셀 크기, 가로 픽셀 크기)

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(8, 5)

for i in range(15):
    ax = axes[i//5, i % 5]
    ax.imshow(x_train[i], cmap='gray')
    ax.axis('off')
    ax.set_title(str(y_train[i]))

plt.tight_layout()
plt.show()

# 정규화
x_train[0, 10:15, 10:15]
print(f'정규화 전 최소값: {x_train.min()}, 최댓값: {x_train.max()}')

x_train = x_train/x_train.max()
print(f'정규화 후 최소값: {x_train.min()}, 최댓값: {x_train.max()}')

x_test = x_test/x_test.max()

# 정규화는 데이터의 전체 범위를 0~1 사이의 값을 가지도록 한다.
# 정규화를 하는 이유는 입력 데이터가 정규화되어 모델이 학습하는 경우 경사하강법 알고리즘에 의한 수렴 속도가 비정규화된 입력 데이터를 가질 때 보다 더 빨리 수렴하기 때문이다. + 국소 최적에 빠지는 현상도 방지한다.

# 정규화를 하더라도 개별 데이터 값의 범위는 축소되지만 원본 배열의 형태는 그대로 유지된다.
# 샘플 이지미즤 형태는 (28,28)로 이루어져 2차원 입력 형태이다. 2차원 입력은 Dense 레이어에 입력값으로 넣을 수 없기 때문에 Dense 레이어에는 입력값으로 반드시 1차원 배열이 들어가야 한다.
print(f'변경 전 shape: {x_train.shape}')
print(f'ID로 shape 변경 후: {x_train.reshape(60000, -1).shape}')

# 또는 flatten 레이어를 사용하여 다차원 데이터를 1차원으로 펼쳐줄 수 있다.
print(f'Flatten 적용 후: {tf.keras.layers.Flatten()(x_train).shape}')


# 활성화 함수: 입력을 비선형 출력으로 변환해주는 함수
# 일반적으로 선형 관계를 나타내는 함수에 비선형성을 추가하는 방법으로 표현된다.
# 비선형성을 추가하지 않고 선형 함수로만 층을 구성한다면 모델을 깊게 구성하더라도 결국 선형함수로 표현된다.

# 시그모이드(Sigmoid)
# sigmoid(x) = 1/(1+e^(-x))

# 하이퍼볼릭 탄젠트(Hyperbolic Tangent/tanh)
# tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

# ReLu(Rectfied Unit)
# ReLu(x) = max(x, 0)
# leaky ReLU(x) = max(x, 0.1x)

# 텐서플로 케라스 레이어에 활성화 함수를 적용하고 싶다면
# tf.keras.layers.Dense(128, activation='relu') # activation 매개변수에 문자열 대입

# 별도의 층처럼 활성화 함수를 대입 -> Batch Normalization(배치 정규화)를 적용한 후 활성화 함수를 적용하고자 할 때 사용
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128),
#     tf.keras.layers.Activation('relu')
# ])

# 딥러닝 모델을 만들 때 첫 번째 레이어에 입력 데이터의 형태를 나타내는 input_shape 매개변수를 지정한다.
# 분류 모델의 가장 마지막 레이어는 출력층으로 출력층의 노드 개수는 반드시 분류해야 할 클래스의 개수와 동일해야 한다. mnist 는 0~9까지의 총 10개의 클래스로 이루어져 있기 때문에 마지막 출력층의 노드 개수는 10개가 되어야 한다.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),  # 마지막 레이어 = 출력층
])
# Dense 레이어를 구성하는 노드의 개수는 조금씩 줄어드는 형태로 구성되었다.
# Flatten 층으로 이미지를 펼치면 784개의 픽셀 값이 각각 입력 변수가 되고 해당 변수에 대한 입력값을 노드 개수를 조금씩 줄여가면서 최종 출력 클래스 개수인 10개까지 정보를 축약하는 것이다.
# 모델의 깊이(=레이어 개수)와 너비(각 레이어를 구성하는 노드의 개수)에 대한 정답은 없고 최적 모델을 찾기 위해선 여러 가지 시도를 통해 최적의 모델 형태를 찾을 필요가 있다.
# 레이어 개수와 노드의 개수도 중요한 하이퍼 파라미터가 된다.

# 출력층의 노드 개수가 2개 이상인 경우 softmax 활성화 함수를 사용한다.
# 다중 분류 문제에서 Softmax 활성화 함수를 사용해야 하지만, 이진 분류 모델의 출력층 노드 개수르 1개로 설정한 경우에는 sigmoid 활성화 함수를 적용한다.
# tf.keras.layers.Dense(1, activation='sigmoid')

# tf.keras.layers.Dense(10, activation='softmax')

# 손실 함수: 모델의 출력층에 따라 올바른 손실함수를 설정해야만 모델이 정상적으로 훈련할 수 있다.
tf.keras.layers.Dense(1, activation='sigmoid')
# 이진 분류기 생성 시 출력층의 노드 개수가 1개면 sigmoid 르 지정하며 손실함수로 binary_crossentropy를 지정하면 된다.
model.compile(loss='binary_crossentropy')

# 출력층의 노드가 2개 이상인 경우 softmax 활성화 함수를 지정하고
# 손실함수는 출력 데이터가 원핫 벡터(one-hot vector) 일 때 categorical_crossentropy로, 원핫 벡터가 아닌 경우 sparse_categorical_crossentropy를 지정한다.
tf.keras.layers.Dense(10, activation='softmax')
model.compile(loss='categorical_crossentropy')

model.compile(loss='sparse_categorical_crossentropy')

# mnist 손글씨 데이터셋은 클래스 개수가 10개이므로 마지막 출력층에 Dense 레이어의 노드 개수를 10으로 지정하고, 출력 데이터가 원핫 벡터가 아닌 0~9까지의 레이블 값을 갖기 때문에 sparse_categorical_crossentropy 를 지정한다.


# 옵티마이저: 손실을 낮추기 위해서 신경망의 가중치와 학습률과 같은 신경망 속성을 변경하는데 사용되는 최적화 방법
# 일반적으로 많이 사용되는 알고리즘은 Adam이며 대체적으로 좋은 성능을 발휘하는 것으로 알려져있다.
# 그밖에 SGD, Adagrad, Nadam, RMSprop, Adadelta, Adamx, Ftrl 등이 있다.

# 옵티마이저는 클래스 인스턴스로 지정
adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam)

# 혹은 문자열로 지정
model.compile(optimizer='adam')
