import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. MNIST 데이터셋 임포트
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('학습 데이터 셋 : ', x_train.shape, y_train.shape)
print('테스트 데이터 셋 : ', y_train.shape, y_test.shape)

# 2. 데이터 전처리
x_train, x_test = x_train/255.0, x_test/255.0

# 3. 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 6. 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print('테스트 정확도:', test_acc)
print('테스트 손실도:', test_loss)

# model.save('MnistModel.h5')

path = 'Img'
count = len(os.listdir(path))
print(count)
for i in range(1, count+1):
    # test.png는 그림판에서 붓으로 숫자 8을 그린 이미지 파일
    # test.png 파일 열어서 L(256단계 흑백이미지)로 변환
    img = Image.open('Img/'+ str(i)+".png").convert("L")

    # 이미지를 784개 흑백 픽셀로 사이즈 변환
    img = np.resize(img, (1, 784))

    # 데이터를 모델에 적용할 수 있도록 가공
    test_data = ((np.array(img) / 255) - 1) * -1

    # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
    res = (model.predict(test_data) > 0.5).astype("int32")

    print('인식된 숫자는 : ', res)