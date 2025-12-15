import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers

class AnomalyGAN:
    def __init__(self, img_shape=(128, 128, 1)):
        self.img_shape = img_shape
        self.latent_dim = 100
        
        # 최적화 함수 설정 (DCGAN 권장 설정: 학습률 0.0002, beta_1 0.5)
        self.optimizer = optimizers.Adam(0.0002, 0.5)

        # 두 명의 플레이어 생성
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # 판별자 컴파일
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # 연결 (GAN)
        self.gan = self.build_gan()

    def build_generator(self):
        # Noise -> 128x128 이미지 생성
        model = tf.keras.Sequential(name="Generator")
        
        # 1. 초기 Dense 레이어 (8x8 크기에서 시작)
        model.add(layers.Dense(8 * 8 * 256, input_dim=self.latent_dim))
        model.add(layers.Reshape((8, 8, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 2. 업샘플링 (8 -> 16)
        model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 3. 업샘플링 (16 -> 32)
        model.add(layers.Conv2DTranspose(64, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 4. 업샘플링 (32 -> 64)
        model.add(layers.Conv2DTranspose(32, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # 5. 최종 출력 (64 -> 128)
        # 픽셀 값은 -1 ~ 1 사이로 맞춤 (tanh)
        model.add(layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh'))
        
        return model

    def build_discriminator(self):
        # 이미지 -> 진짜/가짜 판별
        model = tf.keras.Sequential(name="Discriminator")
        
        # 1. 다운샘플링 (128 -> 64)
        model.add(layers.Conv2D(32, 4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25)) # 과적합 방지

        # 2. 다운샘플링 (64 -> 32)
        model.add(layers.Conv2D(64, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization()) # 판별자에도 배치 정규화 추가
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))

        # 3. 다운샘플링 (32 -> 16)
        model.add(layers.Conv2D(128, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        # 4. 다운샘플링 (16 -> 8)
        model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))

        # 5. 판별 (Flatten -> Dense)
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        
        model = Model(z, validity)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model

    def train(self, normal_images, epochs=2000, batch_size=32):
        # 데이터 정규화 확인 (-1 ~ 1 사이여야 함)
        # normal_images는 이미 전처리 되었다고 가정합니다.
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Discriminator 학습
            # ---------------------
            idx = np.random.randint(0, normal_images.shape[0], batch_size)
            real_imgs = normal_images[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            # 레이블 스무딩: 진짜 이미지를 1.0이 아닌 0.9로 설정하여 D가 너무 강해지는 것 방지
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid * 0.9) 
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Generator 학습
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Generator는 판별자를 속여서 '1'을 뱉게 만들어야 함
            g_loss = self.gan.train_on_batch(noise, valid)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")