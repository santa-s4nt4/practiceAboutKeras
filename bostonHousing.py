import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()

'''
print(train_data.shape)
print(test_data.shape)
print(train_targets)
'''

# データの正規化(特徴量の平均値を引き，標準偏差で割る)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# モデルの定義
def build_model():
    # 同じモデルを複数回インスタンス化する必要があるため，モデルをインスタンス化するための関数を使用．
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# k分割交差検証
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)

    # 検証データの準備 : フォールドiのデータ
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 訓練データの準備 : 残りのフォールドのデータ
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # kerasモデルを構築(コンパイル済み)
    model = build_model()

    # モデルをサイレントモード(verbose=0)で適合
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)

    # モデルを検証データで評価
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))


# フォールドごとに検証ログを保存
num_epochs = 500
all_mae_histories = []
for i in range(k):
    # 検証データの準備 : フォールドiのデータ
    print('processing fold', i)
    val_data = \
        train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = \
        train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 訓練データの準備 : 残りのフォールドのデータ
    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples],
         train_targets[(i + 1) * num_val_samples]],
        axis=0)

    # kerasモデルを構築(コンパイル済み)
    model = build_model()

    # モデルをサイレントモード(verbose=0)で適合
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# k分割交差検証の平均スコアの履歴を構築
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 検証スコアのプロット
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 最初の10個のデータ点を除外した検証スコアのプロット
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
    smooth_mae_history = smooth_curve(average_mae_history[10:])

    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')


plt.ylabel('Validation MAE')
plt.show()

# 最終的なモデルの訓練
# コンパイル済みの新しいモデルの取得
model = build_model()

# データ全体を使って訓練
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)

# テストデータでの検証スコアを取得
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
