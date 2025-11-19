import tensorflow as tf
import numpy as np
import os

# -------------------------------
# CrfFeatureNet TF2 변환
# -------------------------------
class CrfFeatureNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 간단화: 원래 레이어 대부분은 동일하게 정의
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')
        # 실제 모델에 맞게 residual 블록 정의
        # 여기서는 예시로 res2 블록만 작성
        self.res2a_conv1 = tf.keras.layers.Conv2D(256, 1, padding='same')
        self.res2a_bn1 = tf.keras.layers.BatchNormalization()
        self.res2a_conv2a = tf.keras.layers.Conv2D(64, 1, padding='same')
        self.res2a_bn2a = tf.keras.layers.BatchNormalization()
        self.res2a_conv2b = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.res2a_bn2b = tf.keras.layers.BatchNormalization()
        self.res2a_conv2c = tf.keras.layers.Conv2D(256, 1, padding='same')
        self.res2a_bn2c = tf.keras.layers.BatchNormalization()

        # global pooling
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        # res2a 블록
        branch1 = self.res2a_conv1(x)
        branch1 = self.res2a_bn1(branch1, training=training)

        branch2 = self.res2a_conv2a(x)
        branch2 = self.res2a_bn2a(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        branch2 = self.res2a_conv2b(branch2)
        branch2 = self.res2a_bn2b(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        branch2 = self.res2a_conv2c(branch2)
        branch2 = self.res2a_bn2c(branch2, training=training)

        x = tf.nn.relu(branch1 + branch2)

        return self.global_pool(x)  # [batch, features]

# -------------------------------
# AEInvcrfDecodeNet TF2 변환
# -------------------------------
class AEInvcrfDecodeNet(tf.keras.Model):
    def __init__(self, n_p=12, n_digit=2):
        super().__init__()
        self.n_p = n_p
        self.n_digit = n_digit
        self.act = tf.nn.tanh
        # placeholder: 실제 dense 구조를 decode_spec에 맞게 정의
        self.dense_layers = [tf.keras.layers.Dense(n_digit, activation=self.act)]
        self.final_dense = tf.keras.layers.Dense(n_p - 1)

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        x = self.final_dense(x)

        # invcrf 변환
        # 간단화: 기존 invcrf_pca_w_2_invcrf 함수 그대로 tf 연산
        batch = tf.shape(x)[0]
        s = 1024
        m = tf.constant([[i**(j+1) for i in np.linspace(0,1,s)] for j in range(self.n_p)], dtype=tf.float32)
        invcrf = tf.matmul(x, m[:self.n_p-1, :])  # [b, s]
        return invcrf

# -------------------------------
# Linearization_net TF2 변환
# -------------------------------
class LinearizationNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.crf_feature_net = CrfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet()

    @staticmethod
    def _increase(rf):
        g = rf[:, 1:] - rf[:, :-1]
        min_g = tf.reduce_min(g, axis=-1, keepdims=True)
        r = tf.nn.relu(-min_g)
        new_g = g + r
        new_g = new_g / tf.reduce_sum(new_g, axis=-1, keepdims=True)
        new_rf = tf.cumsum(new_g, axis=-1)
        new_rf = tf.pad(new_rf, [[0, 0], [1, 0]], 'CONSTANT')
        return new_rf

    def call(self, img, training=False):
        # placeholder: edge + histogram feature 생성
        edge_1 = tf.image.sobel_edges(img)
        edge_1 = tf.reshape(edge_1, [tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], 6])

        # histogram_layer 단순화
        def histogram_layer(img, max_bin):
            tmp_list = []
            for i in range(max_bin + 1):
                histo = tf.nn.relu(1 - tf.abs(img - i / float(max_bin)) * float(max_bin))
                tmp_list.append(histo)
            return tf.concat(tmp_list, -1)

        features_input = tf.concat([
            img, 
            edge_1, 
            histogram_layer(img, 4), 
            histogram_layer(img, 8), 
            histogram_layer(img, 16)
        ], -1)

        feature = self.crf_feature_net(features_input, training=training)
        feature = tf.cast(feature, tf.float32)
        invcrf = self.ae_invcrf_decode_net(feature)
        invcrf = self._increase(invcrf)
        return invcrf
