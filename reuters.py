from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras import layers
from keras import models
import numpy as np
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))
print(train_data[10])

# ニュースサンプルをテキストに変換する
word_index = reuters.get_word_index()
reverse_word_index = \
    dict([(value, key) for (key, value) in word_index.items()])

# インデックスのオフセットとして3が指定されているのは，0,1,2がそれぞれ"パディング", "シーケンスの開始", "不明"のインデックスとして予約されているためであることに注意する．
decode_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# デコードしたニュースの内容を表示
print(decode_newswire)


# データのエンコーディング
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results


# 訓練データのベクトル化
x_train = vectorize_sequences(train_data)
# テストデータのベクトル化
x_test = vectorize_sequences(test_data)


# one-hotエンコーディング
'''
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# ベクトル化された訓練ラベル
one_hot_train_labels = to_one_hot(train_labels)
# ベクトル化されたテストラベル
one_hot_test_labels = to_one_hot(test_labels)
'''
# ↓ Kerasでone-hotエンコーディング
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# モデルの定義
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 検証データセットの設定
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# モデルの訓練
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# 訓練データと検証データでの損失値をプロット
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 訓練データと検証データでの正解率をプロット
plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# モデルの訓練をやり直す
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

predictions = model.predict(x_test)

predictions[0].shape

np.sum(predictions[0])

np.argmax(predictions[0])

x_train = np.array(train_labels)
x_test = np.array(test_labels)

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy', metrics=['acc'])

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
                    
model.fit(partial_x_train, partial_y_train, epochs=20,
          batch_size=128, validation_data=(x_val, y_val))
