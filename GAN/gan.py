import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers
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

class FastAnomalyGAN:
    def __init__(self, img_shape=(128, 128, 1), learning_rate=0.0002, lr_decay_steps=2000, lr_decay_rate=0.98):
        self.img_shape = img_shape
        self.learning_rate = learning_rate
        
        # 가중치 초기화
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True,
        )
        
        # 손실 함수
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae = tf.keras.losses.MeanAbsoluteError() # L1 Loss (Reconstruction)

        # 옵티마이저
        self.g_optimizer = optimizers.Adam(self.lr_schedule, 0.5, clipnorm=1.0)
        self.d_optimizer = optimizers.Adam(self.lr_schedule, 0.5, clipnorm=1.0)

        # 모델 구축
        self.generator = self.build_autoencoder_generator() # 변경됨: Autoencoder 구조
        self.discriminator = self.build_discriminator()

    def build_autoencoder_generator(self):
        """
        Input: Image (128x128)
        Output: Reconstructed Image (128x128)
        구조: Encoder -> Bottleneck -> Decoder
        Decoder 채널 구성을 AnomalyGAN generator와 맞춰 가중치 이식 가능하도록 조정.
        """
        inputs = layers.Input(shape=self.img_shape)
        
        # --- Encoder (Downsampling) ---
        # 128 -> 64
        e1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=self.weight_init)(inputs)
        e1 = layers.LeakyReLU(0.2)(e1)
        
        # 64 -> 32
        e2 = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=self.weight_init)(e1)
        e2 = layers.BatchNormalization()(e2)
        e2 = layers.LeakyReLU(0.2)(e2)
        
        # 32 -> 16
        e3 = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=self.weight_init)(e2)
        e3 = layers.BatchNormalization()(e3)
        e3 = layers.LeakyReLU(0.2)(e3)
        
        # 16 -> 8 (Bottleneck) — 채널을 256으로 맞춰 AnomalyGAN 디코더와 동일 입력 채널을 사용
        b = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=self.weight_init)(e3)
        b = layers.BatchNormalization()(b)
        b = layers.LeakyReLU(0.2)(b)

        # --- Decoder (Upsampling) ---
        # 8 -> 16
        d1 = layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer=self.weight_init, name="dec_deconv1")(b)
        d1 = layers.BatchNormalization()(d1)
        d1 = layers.Activation('relu')(d1)

        # 16 -> 32
        d2 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer=self.weight_init, name="dec_deconv2")(d1)
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.Activation('relu')(d2)

        # 32 -> 64
        d3 = layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer=self.weight_init, name="dec_deconv3")(d2)
        d3 = layers.BatchNormalization()(d3)
        d3 = layers.Activation('relu')(d3)

        # 64 -> 128
        outputs = layers.Conv2DTranspose(self.img_shape[-1], 4, strides=2, padding='same', activation='tanh', name="dec_out")(d3)

        return Model(inputs, outputs, name="Autoencoder_Generator")

    def build_discriminator(self):
        """
        PatchGAN 스타일 혹은 일반 판별자.
        여기서는 픽셀 단위 디테일을 위해 일반적인 CNN 판별자를 사용합니다.
        """
        model = tf.keras.Sequential(name="Discriminator")
        
        # 128 -> 64
        model.add(layers.Conv2D(64, 4, strides=2, padding='same', input_shape=self.img_shape, kernel_initializer=self.weight_init))
        model.add(layers.LeakyReLU(0.2))
        
        # 64 -> 32
        model.add(layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        
        # 32 -> 16
        model.add(layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        
        # 16 -> 8
        model.add(layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=self.weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.3)) # 드롭아웃 추가
        model.add(layers.Dense(1, activation=None)) # Logits 출력
        
        return model

    @tf.function
    def train_step(self, real_imgs):
        # ---------------------
        #  1. Train Discriminator
        # ---------------------
        with tf.GradientTape() as tape:
            # Generator(Autoencoder)가 이미지를 재구성
            fake_imgs = self.generator(real_imgs, training=True)
            
            real_logits = self.discriminator(real_imgs, training=True)
            fake_logits = self.discriminator(fake_imgs, training=True)
            
            # Label smoothing + noise (Real: U(0.9,1.0), Fake: U(0.0,0.1))
            real_labels = tf.random.uniform(tf.shape(real_logits), 0.9, 1.0)
            fake_labels = tf.random.uniform(tf.shape(fake_logits), 0.0, 0.1)
            d_loss_real = self.bce(real_labels, real_logits)
            d_loss_fake = self.bce(fake_labels, fake_logits)
            
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # ---------------------
        #  2. Train Generator
        # ---------------------
        with tf.GradientTape() as tape:
            fake_imgs = self.generator(real_imgs, training=True)
            fake_logits = self.discriminator(fake_imgs, training=True)
            
            # Adv Loss: 판별자를 속여야 함 (Real로 분류되도록)
            g_adv_loss = self.bce(tf.ones_like(fake_logits), fake_logits)
            
            # Recon Loss (L1): 입력 이미지와 출력 이미지가 같아야 함
            g_recon_loss = self.mae(real_imgs, fake_imgs)
            
            # Structural Loss (SSIM): 구조적 유사성 추가 (Perceptual Loss의 경량화 버전)
            # SSIM은 -1~1 범위이므로 (1 - SSIM) / 2 로 loss화
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(real_imgs * 0.5 + 0.5, fake_imgs * 0.5 + 0.5, max_val=1.0))

            # Total Generator Loss
            # 재구성/구조 비중을 완화해 과도한 블러를 방지
            w_adv = 1.0
            w_recon = 20.0
            w_ssim = 5.0
            
            g_loss = (w_adv * g_adv_loss) + (w_recon * g_recon_loss) + (w_ssim * ssim_loss)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return d_loss, g_loss, g_recon_loss

    def train(self, normal_images, epochs=2000, batch_size=32, save_dir=None, sample_interval=50):
        # 데이터 증강 (Random Flip, Zoom 등)을 tf.data 파이프라인에 추가하면 좋습니다.
        dataset = tf.data.Dataset.from_tensor_slices(normal_images)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print("🚀 학습 시작 (Adversarial Autoencoder 방식)")
        best_recon = np.inf
        
        for epoch in range(epochs):
            start = time.time()
            d_losses = []
            g_losses = []
            recon_losses = []
            
            for batch in dataset:
                # 데이터 증강은 여기서 map 함수로 적용 가능
                d_l, g_l, r_l = self.train_step(batch)
                d_losses.append(d_l)
                g_losses.append(g_l)
                recon_losses.append(r_l)
                
            mean_recon = np.mean(recon_losses)
            if mean_recon < best_recon and save_dir:
                best_recon = mean_recon
                self.generator.save(os.path.join(save_dir, 'best_generator.h5'))
                print(f"Epoch {epoch}: 🔥 best generator saved (Recon {best_recon:.4f})")

            if epoch % sample_interval == 0:
                print(f"Epoch {epoch} [D: {np.mean(d_losses):.4f}] [G: {np.mean(g_losses):.4f}] [Recon L1: {np.mean(recon_losses):.4f}] - {time.time()-start:.2f}s")
                if save_dir:
                    self._save_samples(save_dir, epoch)

        if save_dir:
            self.generator.save(os.path.join(save_dir, 'final_generator.h5'))

    def detect_anomaly(self, test_images):
        """
        빠른 이상 탐지 수행
        """
        # 1. 재구성 (Encoder -> Decoder)
        reconstructed = self.generator.predict(test_images)
        
        # 2. 픽셀 단위 오차 (L1)
        diff = np.abs(test_images - reconstructed)
        
        # 3. 채널 평균 -> 흑백 히트맵 생성
        anomaly_map = np.mean(diff, axis=-1)
        
        # 4. 점수 산출 (이미지 전체 평균 혹은 패치 평균)
        anomaly_score = np.mean(anomaly_map, axis=(1, 2))
        
        return anomaly_score, anomaly_map, reconstructed

    def copy_decoder_from_anomaly_generator(self, anomaly_generator):
        """
        AnomalyGAN generator의 Conv2DTranspose 블록과 필터 구성을 맞춘 상태에서
        가중치를 디코더에만 이식합니다. 필터/커널 형상이 맞는 층만 복사합니다.
        """
        src = [l for l in anomaly_generator.layers if isinstance(l, layers.Conv2DTranspose)]
        dst = [l for l in self.generator.layers if isinstance(l, layers.Conv2DTranspose)]
        copied = 0
        for s, d in zip(src, dst):
            sw = s.get_weights()
            dw = d.get_weights()
            if not sw or not dw:
                continue
            if sw[0].shape == dw[0].shape and sw[-1].shape == dw[-1].shape:
                d.set_weights(sw)
                copied += 1
        print(f"Decoder weight transfer: {copied}/{len(dst)} layers matched")

    def _save_samples(self, save_dir, epoch, num=9):
        # 여기서는 입력이 아니라 노이즈로 생성하지 않으므로, 샘플 대신 재구성 데모를 저장하지 않음
        # 간단히 현재 가중치로 self.generator의 출력을 저장 (zero input 기준)
        dummy = tf.zeros((num,) + self.img_shape)
        imgs = self.generator(dummy, training=False).numpy()
        imgs = 0.5 * imgs + 0.5
        side = int(np.ceil(np.sqrt(num)))
        import matplotlib.pyplot as plt
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

