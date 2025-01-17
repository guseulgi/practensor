# 모듈
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


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


# 평가지표
# 분류 모델에 대한 평가지표(metrics)는 정확도를 나타내는 accuracy(acc) 가 가장 많이 사용되며, auc, precision, recall 등의 지표로도 사용된다.

# 모델 컴파일 단계에서 metrics 매개변수에 파이썬 리스트 형태로 하나 이상의 평가지표를 지정하여 여러 지표들을 동시에 참고할 수 있다.
# 클래스 인스턴스로 지정
acc = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=[acc])

# 문자열로 지정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 검증 데이터셋
# 모델을 훈련할 때 검증셋을 추가 지정하면 매 epoch 마다 훈련 손실과 검증 손실, 각 셋에 대한 평가지표를 나란히 출력한다.
# validation_data 매개변수에 튜플 형식으로 검증 셋을 지정해준다.
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=10)

# 평가
# 훈련이 종료되면 evaluate() 메서드로 모델 성능을 검증하고 평가 결과를 확인할 수 있다.
test_loss, test_acc = model.evaluate(x_test, y_test)
print('검증 셋 정확도', test_acc)


# 예측
# predict() 메서드에 이미지 데이터를 넣어주면 모델의 예측 결과를 반환한다. 예측한 분류 결과를 넘파이 배열 형태로 저장
predictions = model.predict(x_test)
predictions[0]  # 예측 결과 출력

print(np.argmax(predictions[0]))
# 가장 높은 확률값을 가진 클래스가 최종 예측된 클래스로 넘파이 배열의 argmax 를 활용하여 가장 높은 확률값을 가지는 클래스 결과를 확인할 수 있다.


def get_one_result(idx):
    img, y_true, y_pred, confidence = x_test[idx], y_test[idx], np.argmax(
        predictions[idx]), 100*np.max(predictions[idx])
    return img, y_true, y_pred, confidence


fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 10)
for i in range(15):
    ax = axes[i//5, i % 5]
    img, y_true, y_pred, confidence = get_one_result(i)
    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'True: {y_true}')
    ax.set_xlabel(f'Prediction: {y_pred}\n Confidence: ({confidence:.2f} %)')
plt.tight_layout()
plt.show()

# 레이어
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('train set: ', x_train.shape, y_train.shape)

x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

# dense = tf.keras.layers.Dense(256, activation='relu')

# 문자열 레이어 초기화
dense = tf.keras.layers.Dense(
    256, kernel_initializer='he_normal', activation='relu')

# 클래스 인스턴스 레이어 초기화
# he_normal = tf.keras.initializers.HeNormal()
# dense = tf.keras.Dense(256, kernel_initializer=he_normal, activation='relu')

dense.get_config()['kernel_initializer']

# 케라스 가중치 초기화
# glorot_normal, glorot_uniform: 글로럿 초기화 (Xavier 초기화)
# lecun_normal, lecun_uniform: Yann Lecun 초기화
# he_normal, he_uniform: He 초기화
# random_normal, random_unoform: 정규분포, 연속균등 분포 초기화

# 규제 kernel_regularizer
dense = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1')

# regulatizer = tf.keras.regularizers.l1(l1=0.1)
# dense = tf.keras.layers.Dense(256, kernel_regularizer=regulatizer, activation='relu')

dense.get_config()

# 드롭아웃
# 딥러닝 모델의 층이 넓고 깊어질 때 모델은 훈련에 주어진 샘플에 과하게 적합하도록 학습됨
# 훈련할 때 만나지 못한 새로운 데이터에 대해 좋지 않은 예측력을 보여서 일반화된 성능을 갖지못하는 문제 = 과대적합
# 노드의 일부 신호를 임의로 삭제하게 되면 학습하는 가중치 파라미터의 개수가 현저하게 줄어들어 모델이 쉽게 과대적합 되는 것을 방지할 수 있다

# 모델이 훈련할 때 드롭아웃이 적용되어 노드 중 일부만 훈련하게 되지만 예측 시점에는 모든 노드들이 활용됨
tf.keras.layers.Dropout(0.25)  # 비율 25%의 노드가 삭제

# 배치 정규화
# 각 층에 활성화 함수를 통과하기 전 미니 배치의 스케일을 정규화한다
# 다음층으로 데이터가 전달되기 전에 스케일을 조정하기에 안정적인 훈련이 가능하고 성능을 향상시킬 수 있음

# Dense + ReLU
model_a = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_a.summary()

# Dense + 배치 정규화 + ReLU
model_b = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.Activation('relu'),  # 배치 정규화 후 활성화 함수

    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])
model_b.summary()


# 활성화 함수
# tf.keras.layers.LeakyReLU()
# tf.keras.layers.LeakyReLU(alpha=0.2)

# Dense + 배치 정규화 + LeakyReLU(0.2)
model_c = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.LeakyReLU(alpha=0.2),  # 배치 정규화 후 활성화 함수

    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),

    tf.keras.layers.Dense(10, activation='softmax')
])
model_c.summary()

model_a.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_b.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_c.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_a = model_a.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=10)
history_b = model_b.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=10)
history_c = model_c.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=10)

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 11),
         history_a.history['val_loss'], color='navy', linestyle=':')
plt.plot(np.arange(1, 11),
         history_b.history['val_loss'], color='tomato', linestyle='-.')
plt.plot(np.arange(1, 11),
         history_c.history['val_loss'], color='green', linestyle='-')

plt.title('Losses', fontsize=20)
plt.xlabel('epochs')
plt.ylabel('Losses')
plt.legend(['ReLU', 'BatchNorm + ReLU', 'batchnorm + LeakyReLU'], fontsize=12)
plt.show()

# 콜백
# 모델을 훈련할 때 보조적인 작업을 수행할 수 있도록 도와주는 객체
# fit() 메서드에 매개변수로 지정 가능

# 모델 체크포인트
# 모델 훈련 시 콜백으로 지정하여 epoch별 모델의 가중치를 저장
# 저장할 때 마치 체크포인트를 생성하듯 미리 정해 놓은 규칙에 의해 체크포인트를 생성하고 저장

# filepath: 체크 포인트의 저장 경로
# save_weights_only: 가중치만 저장할지 여부 설정
# save_best_only: monitor 기준으로 가장 높은 epoch만 저장할지 아니면 매 epoch별 저장할지 여부 설정
# monitor: 저장 시 기준이 되는 지표 설정 val_loss 로 지정 시 검증 손실이 가장 낮은 epoch의 가중치로 저장
# verbose: 1로 설정 시 매 epoch별 저장 여부를 알려주는 로그 메시지 출력

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='tmp_checkpoint.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
model_a.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=10, callbacks=[checkpoint])

# 저장한 가중치를 model 인스턴스에 적용하려면 load_weights() 메서드에 모델 체크포인트  파일 경로를 지정하여 호출해 주어야 한다
# 모델에 저장한 가중치를 명시적으로 로드해주어야 검증 손실이 가장 낮았던 가중치가 모델에 로드됨


# 모델 체크 로드 전
loss, acc = model_a.evaluate(x_test, y_test)
print(f'체크 포인트 로드 전: , {loss:3f}, {acc:3f}')

model_a.load_weights('tmp_checkpoint.weights.h5')
loss, acc = model_a.evaluate(x_test, y_test)
print(f'체크 포인트 로드 후: , {loss:3f}, {acc:3f}')
# load_weights() 메서드가 호출되기 전 검증 성능은 마지막 epoch 10 실행 결과로 손실 기준이다
# 모델 체크 포인트에 저장되어 있는 가중치를 불러온 뒤에는 가장 검증 손실이 낮았던 epoch의 손실이 된다 (epoch 5)

# 조기 종료
# tensorflow.keras.callbacks.EarlyStopping() 객체로 생성하며 모델 훈련 시 patience에 지정된 epoch 안에 손실이 줄어들지 않는다면 모델 훈련을 조기 종료
# 조기 종료 기준이 되는 지표를 검증 손실로 설정하고 조기 종료 콜백을 생성한다
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3)

model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=20, callbacks=[earlystopping])

# 학습률 스케줄러
# tensorflow.keras.callbacks.LearningRateScheduler() 객체로 생성하여 훈련에 대한 학습률을 조정


def scheduler(epoch, lr):
    tf.print(f'learning_rate: {lr:5f}')
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.1)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(tf.keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(round(model.optimizer.lr.numpy(), 5))

# 텐서보드
# 훈련에 대한 시각화를 실시간으로 제공
# epoch별 훈련 손실 및 평가 지표를 시각화해 차트로 보여주고 모델의 구조를 시각화해 보여주거나 레이어의 가중치 분포도를 시각화로 제공
log_dir = 'tensorboard'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=10, callbacks=[tensorboard])
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[lr_scheduler])

# round(model.optimizer.lr.numpy(), 5)

# 텐서보드 출력
# %load_ext tensorboard
# %tensorboard --logdir {log_dir}


# 모델을 h5 포맷으로 저장
model.save('h5-model.h5')
# HDF5: Hierarchical Data Format, 대용량의 데이터를 저장하기 위한 파일 형식
# SavedModel

# 저장된 모델 복원
h5_model = tf.keras.models.load_model('h5-model.h5')
h5_model.summary()

# 함수형 API
# 1.Input 레이어 정의 - 데이터의 입력 shape 를 지정
input_layer = tf.keras.Input(shape=(28, 28), name='InputLayer')

# 레이어마다 반환되는 출력 값을 변수에 저장한 뒤 다음 레이어의 입력으로 연결
# 여러 개의 레이어를 체인 구조로 입출력을 연결
x1 = tf.keras.layers.Flatten(name='Flatten')(input_layer)
x2 = tf.keras.layers.Dense(256, activation='relu', name='Dense1')(x1)
x3 = tf.keras.layers.Dense(64, activation='relu', name='Dense2')(x2)
x4 = tf.keras.layers.Dense(10, activation='softmax', name='OutputLayer')(x3)

func_model = tf.keras.Model(
    inputs=input_layer, outputs=x4, name='FunctionalModel')
# inputs 매개변수로 입력층 지정
# outputs 매개변수로 출력층 지정
func_model.summary()

plot_model(func_model, show_shapes=True,
           show_layer_names=True, to_file='model.png')

func_model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

func_model.fit(x_train, y_train, epochs=3)
loss, acc = func_model.evaluate(x_test, y_test, verbose=0)
