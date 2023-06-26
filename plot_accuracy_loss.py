# import thư viện
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as pyplot

# Thực hiện load dữ liệu
iris_data = load_iris() 

# In ra 10 input đầu tiên
print('First 10 inputs: ')
print(iris_data.data[:10])
# In ra 10 output đầu tiên
print('First 10 output (label): ')
print(iris_data.target[:10])

# Gán input vào biến X
X = iris_data.data
# Gán output vào biến y 
y = iris_data.target.reshape(-1,1)

# Thực hiện Onehot transform
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
print("Output after transform")
print(y)

# Chia dữ liệu train, test với tỷ lệ 80% cho train và 20% cho test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Khai báo model
model = Sequential()

model.add(Dense(128, input_shape=(4,), activation='relu', name='layer1'))
model.add(Dense(128, activation='relu', name='layer2'))
model.add(Dense(3, activation='softmax', name='output'))

# Cài đặt hàm tối ưu Adam 
optimizer = Adam()
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# In cấu trúc mạng ra màn hình
print('Detail of network: ')
print(model.summary())

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Kiểm tra trên tập test
results = model.evaluate(X_test, y_test)
print('Test loss: {:4f}'.format(results[0]))
print('Test accuracy: {:4f}'.format(results[1]))

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=200,validation_data=(X_test,y_test))

# plot loss và accuracy
pyplot.figure(figsize=(20,10))
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

