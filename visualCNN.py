import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.applications import VGG16
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())

# 5.2.2項でsmallデータセットを格納したディレクトリへのパスであることに注意．
img_path = 'cats_and_dogs_small/test/cats/cat.1700.jpg'

# この画像を4次元テンソルとして前処理
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# このモデルの訓練に使用された入力が次の方法で前処理されていることに注意．
img_tensor /= 255.

# 形状は(1, 150, 150, 3)
print(img_tensor.shape)

# テスト画像を表示
plt.imshow(img_tensor[0])
plt.show

# 入力テンソルと出力テンソルのリストに基づいてモデルをインスタンス化
# 出力側の8つの層から出力を抽出
layer_outputs = [layer.output for layer in model.layers[:8]]

# 特定の入力を元にこれらの出力を返すモデルを作成
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 5つのNumPy配列(層の活性化ごとに1つ)のリストを返す
activations = activation_model.predict(img_tensor)

# e.g. 猫の入力画像に対する最初の畳み込み層の活性化↓
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 3番目のチャンネルを可視化
# plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
# plt.show()

# 30番目のチャンネルを可視化
# plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
# plt.show()

# 中間層の活性化ごとにすべてのチャンネルを可視化
# プロットの一部として使用する層の名前
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# 特徴マップを表示
for layer_name, layer_activation in zip(layer_names, activations):
    # 特徴マップに含まれている特徴量の数
    n_features = layer_activation.shape[-1]

    # 特徴マップの形状(1, size, size, n_features)
    size = layer_activation.shape[1]

    # この行列で活性化のチャンネルをタイル表示
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 各フィルタを1つの大きな水平グリッドでタイル表示
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row]
            # 特徴量の見た目を良くするための後処理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
                         row * size: (row + 1) * size] = channel_image

    # グリッドを表示
    '''
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    '''

# フィルタを可視化するための損失テンソルの定義
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# gradientsの呼び出しはテンソル(kの場合はサイズ1)のリストを返す
# このため，最初の要素(テンソル)だけを保持する
grads = K.gradients(loss, model.input)[0]

# 除算の前に1e-5を足すことで，0による除算を回避
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 入力値をNumPy配列で受け取り，出力値をNumPy配列で返す配列
iterate = K.function([model.input], [loss, grads])

# テストしてみる
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 確率的勾配降下法を使って損失値を最大化
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
# 勾配上昇法を40ステップ実行
step = 1.  # 各勾配の更新の大きさ
for i in range(40):
    # 損失値と勾配値を計算
    loss_value, grads_value = iterate([input_img_data])
    # 損失が最大になる方向に入力画像を調整
    input_img_data += grads_value * step

# テンソルを有効な画像に変換するユーティリティ関数


def deprocess_image(x):
    # テンソルを正規化 : 中心を0，標準偏差を0.1にする
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1]でクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# フィルタを活性化するための関数


def generate_pattern(layer_name, filter_index, size=150):
    # ターゲット層のn番目のフィルタの活性化を最大化する損失関数を構築
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # この損失関数を使って入力画像の勾配を計算
    grads = K.gradients(loss, model.input)[0]

    # 正規化トリック : 勾配を正規化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて損失値と勾配値を返す関数
    iterate = K.function([model.input], [loss, grads])

    # 最初はノイズが含まれたグレースケール画像を試用
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 勾配上昇法を40ステップ実行
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


'''
plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()
'''

# 層の各フィルタの応答パターンで構成されたグリッドの生成
layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
for layer_name in layers:
    size = 64
    margin = 5

    # 結果を格納する空(黒)の画像
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):  # resultsグリッドの行を順番に処理
        for j in range(8):  # resultsグリッドの列を順番に処理
            # layer_nameのフィルタi + (j * 8)のパターンを生成
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # resultsグリッドの矩形(i, j)に結果を配置
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start:horizontal_end,
                    vertical_start: vertical_end, :] = filter_img

# resultsグリッドを表示
'''
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()
'''

# 出力側に全結合分類器が含まれていることに注意
# ここまでのケースでは，この分類機を削除している
model = VGG16(weights='imagenet')

# ターゲット画像へのローカルパス
img_path = '/Users/fchollet/Downloads/creative_commons_elephant.jpg'
# ターゲット画像を読み込む:imgはサイズが224×224のPIL画像
img = image.load_img(img_path, target_size=(224, 224))
# xは形状が(224, 224, 3)のfloat32型のNumPy配列
x = image.img_to_array(img)
# この配列をサイズが(1, 224, 224, 3)のバッチに変換するために次元を追加
x = np.expand_dims(x, axis=0)
# バッチの前処理(チャネルごとに色を正規化)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])
np.argmax(preds[0])

# Grad-CAM アルゴリズムの設定
# 予測ベクトルの「アフリカゾウ」エントリ
african_elephant_output = model.output[:, 386]
# VGG16の最後の畳み込み層であるblock5_conv3の出力特徴マップ
last_conv_layer = model.get_layer('block5_conv3')
# block5_conv3の出力特徴マップでの「アフリカゾウ」クラスの勾配
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# 形状が(512,)のベクトル:
# 各エントリは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# 2頭のアフリカゾウのサンプル画像に基づいて、pooled_gradsと
# block5_conv3の出力特徴マップの値にアクセスするための関数
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
# これら2つの値をNumPy配列として取得
pooled_grads_value, conv_layer_output_value = iterate([x])
# 「アフリカゾウ」クラスに関する「このチャネルの重要度」を
# 特徴マップ配列の各チャネルに掛ける
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 最終的な特徴マップのチャネルごとの平均値が
# クラスの活性化のヒートマップ
heatmap = np.mean(conv_layer_output_value, axis=-1)

# ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# ヒートマップを元の画像にスーパーインポーズ
# cv2を使って元の画像を読み込む
img = cv2.imread(img_path)
# 元の画像と同じサイズになるようにヒートマップのサイズを変更
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)
# ヒートマップを元の画像に適用
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 0.4はヒートマップの強度係数
superimposed_img = heatmap * 0.4 + img
# 画像をディスクに保存
cv2.imwrite('elephant_cam.jpg', superimposed_img)
