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

#
