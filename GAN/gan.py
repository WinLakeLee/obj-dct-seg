import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import VGG19
import time
import os

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except Exception:
    HAS_TFA = False

class AnomalyGAN:
    def __init__(self, img_shape=(128, 128, 1), latent_dim=100, learning_rate=0.0002, beta_1=0.5, dropout_rate=0.2, spectral_norm=False):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.dropout_rate = dropout_rate
        self.spectral_norm = spectral_norm and HAS_TFA
        if spectral_norm and not HAS_TFA:
            print("⚠️ tensorflow_addons가 없어 spectral_norm 옵션을 무시합니다. `pip install tensorflow-addons` 후 다시 시도하세요.")

        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # 최적화 함수 설정 (DCGAN 권장 설정: 학습률 0.0002, beta_1 0.5) + gradient clipping
        self.g_optimizer = optimizers.Adam(self.learning_rate, self.beta_1, clipnorm=1.0)
        self.d_optimizer = optimizers.Adam(self.learning_rate, self.beta_1, clipnorm=1.0)

        # 두 명의 플레이어 생성
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # 연결 (GAN) - from_logits=True라 판별자 출력은 시그모이드 없이 사용
        self.gan = self.build_gan()

    def build_generator(self):
        # Noise -> HxW 이미지 생성
        model = tf.keras.Sequential(name="Generator")

        # 시작 해상도는 입력 해상도의 1/16 (예: 128 -> 8)
        start_dim = self.img_shape[0] // 16
        start_channels = 256

        # 1. 초기 Dense 레이어
        model.add(layers.Dense(start_dim * start_dim * start_channels, input_dim=self.latent_dim, kernel_initializer=self.weight_init))
        model.add(layers.Reshape((start_dim, start_dim, start_channels)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 2. 업샘플링 (8 -> 16)
        model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 3. 업샘플링 (16 -> 32)
        model.add(layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 4. 업샘플링 (32 -> 64)
        model.add(layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 5. 최종 출력 (64 -> 128)
        # 픽셀 값은 -1 ~ 1 사이로 맞춤 (tanh)
        model.add(layers.Conv2DTranspose(self.img_shape[-1], 4, strides=2, padding='same', activation='tanh', kernel_initializer=self.weight_init))
        
        return model

    def build_discriminator(self):
        # 이미지 -> 진짜/가짜 판별
        model = tf.keras.Sequential(name="Discriminator")

        def conv(filters):
            layer = layers.Conv2D(filters, 4, strides=2, padding='same', kernel_initializer=self.weight_init)
            if self.spectral_norm:
                return tfa.layers.SpectralNormalization(layer)
            return layer

        # 1. 다운샘플링 (128 -> 64)
        model.add(conv(32))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(self.dropout_rate))

        # 2. 다운샘플링 (64 -> 32)
        model.add(conv(64))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(self.dropout_rate))

        # 3. 다운샘플링 (32 -> 16)
        model.add(conv(128))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(self.dropout_rate))
        
        # 4. 다운샘플링 (16 -> 8)
        model.add(conv(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(self.dropout_rate))

        # 5. 판별 (Flatten -> Dense) - logits 출력
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation=None, kernel_initializer=self.weight_init))
        
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        
        model = Model(z, validity)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=self.g_optimizer)
        return model

    def train(self, normal_images, epochs=2000, batch_size=32, buffer_size=1000, label_flip_prob=0.02):
        """
        normal_images: numpy array or tf tensor, values expected in [-1, 1].
        Uses tf.data + GradientTape for faster, stabler training.
        """
        dataset = tf.data.Dataset.from_tensor_slices(normal_images)
        dataset = dataset.shuffle(min(buffer_size, normal_images.shape[0])).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        @tf.function
        def train_step(real_imgs):
            b = tf.shape(real_imgs)[0]
            noise = tf.random.normal((b, self.latent_dim))

            with tf.GradientTape(persistent=True) as tape:
                fake_imgs = self.generator(noise, training=True)
                real_logits = self.discriminator(real_imgs, training=True)
                fake_logits = self.discriminator(fake_imgs, training=True)

                real_labels = tf.random.uniform(tf.shape(real_logits), 0.9, 1.0)
                fake_labels = tf.random.uniform(tf.shape(fake_logits), 0.0, 0.1)

                if label_flip_prob > 0.0:
                    flip_real = tf.cast(tf.random.uniform(tf.shape(real_logits)) < label_flip_prob, tf.float32)
                    flip_fake = tf.cast(tf.random.uniform(tf.shape(fake_logits)) < label_flip_prob, tf.float32)
                    real_labels = real_labels * (1.0 - flip_real) + fake_labels * flip_real
                    fake_labels = fake_labels * (1.0 - flip_fake) + real_labels * flip_fake

                d_loss_real = self.bce(real_labels, real_logits)
                d_loss_fake = self.bce(fake_labels, fake_logits)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                g_loss = self.bce(tf.ones_like(fake_logits), fake_logits)

            d_vars = self.discriminator.trainable_variables
            g_vars = self.generator.trainable_variables
            d_grads = tape.gradient(d_loss, d_vars)
            g_grads = tape.gradient(g_loss, g_vars)
            self.d_optimizer.apply_gradients(zip(d_grads, d_vars))
            self.g_optimizer.apply_gradients(zip(g_grads, g_vars))
            return d_loss, g_loss

        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            for batch in dataset:
                d_loss, g_loss = train_step(batch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

            if epoch % 50 == 0:
                d_val = tf.reduce_mean(d_losses)
                g_val = tf.reduce_mean(g_losses)
                print(f"Epoch {epoch} [D loss: {d_val:.4f}] [G loss: {g_val:.4f}]")

    # Provide a simple sample-saving helper for DCGAN-style training loops
    def save_sample_images(self, save_dir, epoch, num=9):
        os.makedirs(save_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        dummy = tf.random.normal((num, self.latent_dim))
        imgs = self.generator(dummy, training=False).numpy()
        imgs = 0.5 * imgs + 0.5
        side = int(np.ceil(np.sqrt(num)))
        fig, axs = plt.subplots(side, side, figsize=(6,6))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < num:
                    img = imgs[idx]
                    if self.img_shape[-1]==1:
                        axs[i,j].imshow(img[:,:,0], cmap='gray')
                    else:
                        axs[i,j].imshow(img)
                axs[i,j].axis('off')
                idx += 1
        fig.savefig(os.path.join(save_dir, f'samples_epoch{epoch}.png'))
        plt.close(fig)

"""
FastAnomalyGAN의 기능을 통합하려면 AnomalyGAN을 확장하여 사용하십시오.
FastAnomalyGAN 클래스는 제거되었습니다.
"""

# 하위 호환용 alias
FastAnomalyGAN = AnomalyGAN

