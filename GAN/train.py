import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# ==========================================
# 1. 모델 클래스 정의 (DCGAN 구조)
# ==========================================
class AnomalyGAN:
    def __init__(self, input_shape, latent_dim, learning_rate):
        self.img_shape = input_shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        self.optimizer = optimizers.Adam(self.learning_rate, 0.5)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential(name="Generator")
        # 입력 차원 계산 (128x128 기준 8x8에서 시작)
        start_dim = self.img_shape[0] // 16 
        
        model.add(layers.Dense(start_dim * start_dim * 256, input_dim=self.latent_dim))
        model.add(layers.Reshape((start_dim, start_dim, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        # Upsampling block
        for filters in [128, 64, 32]:
            model.add(layers.Conv2DTranspose(filters, 4, strides=2, padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))

        # Final output
        model.add(layers.Conv2DTranspose(self.img_shape[-1], 4, strides=2, padding='same', activation='tanh'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential(name="Discriminator")
        
        model.add(layers.Conv2D(32, 4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))

        for filters in [64, 128, 256]:
            model.add(layers.Conv2D(filters, 4, strides=2, padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=0.2))
            model.add(layers.Dropout(0.25))

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

    def save_sample_images(self, epoch, save_dir='images'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale to [0, 1] for plot

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.img_shape[-1] == 1: # Grayscale
                    axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                else:
                    axs[i,j].imshow(gen_imgs[cnt, :, :])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f"{save_dir}/epoch_{epoch}.png")
        plt.close()

# ==========================================
# 2. 데이터 로드 함수 (사용자 데이터 연결부)
# ==========================================
def load_data(img_shape):
    # TODO: 여기에 실제 데이터를 로드하는 코드를 작성하세요.
    # 현재는 테스트를 위해 랜덤 노이즈 데이터를 생성합니다.
    print("⚠️ 경고: 실제 데이터가 연결되지 않았습니다. 테스트용 랜덤 데이터를 사용합니다.")
    X_train = np.random.normal(0, 1, (1000, img_shape[0], img_shape[1], img_shape[2]))
    
    # 데이터 정규화 (-1 ~ 1)
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    return X_train

# ==========================================
# 3. 학습 및 실행 로직
# ==========================================
def train():
    # 파라미터 파싱 (외부에서 변수 조정 가능하도록 설정)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=0.0002, help='학습률 (Learning Rate)')
    parser.add_argument('--latent_dim', type=int, default=100, help='잠재 공간 차원')
    parser.add_argument('--interval', type=int, default=100, help='이미지 저장 및 로그 출력 간격')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='모델 저장 경로')
    args = parser.parse_args()

    # 설정값 출력
    print(f"\n🚀 학습 시작! 설정값: {args}\n")

    # 디렉토리 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 모델 초기화
    img_shape = (128, 128, 1) # 필요시 (128, 128, 3)으로 변경
    gan = AnomalyGAN(img_shape, args.latent_dim, args.lr)
    
    # 데이터 로드
    X_train = load_data(img_shape)

    # 학습 루프용 변수
    valid = np.ones((args.batch_size, 1))
    fake = np.zeros((args.batch_size, 1))
    
    # **가장 이상적인 모델을 찾기 위한 변수**
    best_g_loss = float('inf') 

    for epoch in range(args.epochs):
        # 1. Discriminator 학습
        idx = np.random.randint(0, X_train.shape[0], args.batch_size)
        imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        gen_imgs = gan.generator.predict(noise)
        
        d_loss_real = gan.discriminator.train_on_batch(imgs, valid * 0.9) # Label Smoothing
        d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 2. Generator 학습
        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        g_loss = gan.gan.train_on_batch(noise, valid)

        # 3. 로깅 및 Best Model 저장
        # 이상 탐지에서는 G loss가 낮은 것이 (보통) 정상 데이터를 잘 흉내낸다는 뜻
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            gan.generator.save(f"{args.save_dir}/best_generator.h5")
            print(f"Epoch {epoch}: 🔥 새로운 Best Model 저장됨! (G Loss: {g_loss:.4f})")

        if epoch % args.interval == 0:
            print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss:.4f}]")
            gan.save_sample_images(epoch)

    # 학습 완료 후 최종 모델 저장
    gan.generator.save(f"{args.save_dir}/final_generator.h5")
    print("\n✅ 학습 완료! 'best_generator.h5'와 'final_generator.h5'가 저장되었습니다.")

if __name__ == '__main__':
    train()