import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import os
import json
import shutil
from datetime import datetime
import argparse
from .gan import AnomalyGAN
from pathlib import Path
import config

from common.data_utils import find_first_class, load_numpy_images
import matplotlib.pyplot as plt

# ==========================================
# 1. 모델 클래스 정의 (DCGAN 구조)
# ==========================================


# ==========================================
# 2. 데이터 로드 함수 (사용자 데이터 연결부)
# ==========================================
def load_data(img_shape, data_dir=None, max_images=None):
    """
    data_dir에 있는 이미지를 로드하여 (N, H, W, C) 형태의 numpy 배열로 반환합니다.
    - 이미지 크기는 img_shape에 맞게 리사이즈됩니다.
    - 채널 수가 1이면 grayscale로 변환합니다.
    - 픽셀값은 -1 ~ 1 범위로 정규화됩니다.
    """
    if not data_dir:
        print("⚠️ 경고: --data_dir가 지정되지 않았습니다. 테스트용 랜덤 데이터를 사용합니다.")
        return np.random.normal(0, 1, (1000, img_shape[0], img_shape[1], img_shape[2]))

    try:
        return load_numpy_images(data_dir, img_shape, max_images=max_images)
    except RuntimeError as e:
        print(f"⚠️ 경고: {e}. 랜덤 데이터를 사용합니다.")
        return np.random.normal(0, 1, (1000, img_shape[0], img_shape[1], img_shape[2]))


def collect_train_images(root, class_name, size=(128, 128), channels=1):
    """Load training images for a class under a given root using shared loader."""
    data_dir = Path(root) / class_name / 'train'
    img_shape = (size[0], size[1], channels)
    return load_numpy_images(data_dir, img_shape)

# ==========================================
# 3. 학습 및 실행 로직
# ==========================================
def run_training(args):
    # 설정값 출력
    print(f"\n🚀 학습 시작! 설정값: {args}\n")

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    img_shape = (args.img_size, args.img_size, args.channels)
    gan = AnomalyGAN(img_shape, args.latent_dim, args.lr)
    # Ensure discriminator is compiled before calling `train_on_batch`.
    # AnomalyGAN builds `gan.gan` (generator+frozen discriminator) and optimizers,
    # but the standalone discriminator needs to be compiled for `train_on_batch`.
    try:
        gan.discriminator.compile(optimizer=gan.d_optimizer, loss=gan.bce)
    except Exception:
        # If compilation fails for any reason, print a helpful message and re-raise.
        print("Failed to compile discriminator; ensure optimizer and loss are valid.")
        raise
    X_train = load_data(img_shape, args.data_dir, args.max_images)

    valid = np.ones((args.batch_size, 1))
    fake = np.zeros((args.batch_size, 1))
    best_g_loss = float('inf')
    g_loss_history = []
    # advanced patience parameters
    patience = getattr(args, 'patience', 5)
    min_delta = getattr(args, 'min_delta', 1e-4)
    min_epochs = getattr(args, 'min_epochs', 10)
    stag_w = getattr(args, 'stagnation_window', 5)
    max_ratio = getattr(args, 'max_improve_ratio', 2.0)
    bonus_epochs = getattr(args, 'bonus_epochs_on_large_improve', 3)

    no_improve = 0
    bonus_remaining = 0

    epoch = 0
    max_epochs = getattr(args, 'epochs', 0)
    unlimited = (max_epochs <= 0)

    while unlimited or epoch < max_epochs:
        epoch += 1
        idx = np.random.randint(0, X_train.shape[0], args.batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        gen_imgs = gan.generator.predict(noise)

        d_loss_real = gan.discriminator.train_on_batch(imgs, valid * 0.9)
        d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        g_loss = gan.gan.train_on_batch(noise, valid)

        improved = False
        g_loss_history.append(float(g_loss))

        # Direct improvement check (absolute improvement)
        if float(g_loss) + min_delta < best_g_loss:
            best_g_loss = float(g_loss)
            gan.generator.save(f"{args.save_dir}/best_generator.h5")
            print(f"Epoch {epoch}: 🔥 새로운 Best Model 저장됨! (G Loss: {float(g_loss):.6f})")
            improved = True
            no_improve = 0
        else:
            # If we haven't reached minimum epochs, don't count as failure
            if epoch < min_epochs:
                no_improve = 0
            else:
                # Sliding-window stagnation detection:
                # Compare the average G-loss of the previous window vs current window.
                # If current_avg + min_delta < prev_avg -> treat as improvement and reset counter.
                if len(g_loss_history) >= 2 * stag_w and stag_w > 0:
                    prev_avg = float(np.mean(g_loss_history[-2 * stag_w:-stag_w]))
                    curr_avg = float(np.mean(g_loss_history[-stag_w:]))
                    if curr_avg + min_delta < prev_avg:
                        # improvement detected in the sliding window
                        no_improve = 0
                        # Large improvement handling: grant bonus epochs
                        if prev_avg / max(curr_avg, 1e-12) > max_ratio:
                            print(f"Epoch {epoch}: Large improvement detected (ratio {prev_avg/curr_avg:.2f}), granting bonus {bonus_epochs} epochs")
                            bonus_remaining = max(bonus_remaining, bonus_epochs)
                    else:
                        # no improvement within the sliding window -> increment
                        no_improve += 1
                else:
                    # Not enough history yet -> increment conservatively
                    no_improve += 1

        # Consume bonus if present (bonus prevents early stopping while positive)
        if bonus_remaining > 0:
            effective_no_improve = 0
            bonus_remaining -= 1
        else:
            effective_no_improve = no_improve

        # Always print a short epoch summary so `no_improve` visibility is immediate.
        try:
            d_loss_val = float(d_loss[0])
        except Exception:
            d_loss_val = float(d_loss)
        g_loss_val = float(g_loss)
        # diagnostic info for sliding-window when available
        win_info = ""
        if stag_w > 0 and len(g_loss_history) >= 2 * stag_w:
            prev_avg = float(np.mean(g_loss_history[-2 * stag_w:-stag_w]))
            curr_avg = float(np.mean(g_loss_history[-stag_w:]))
            win_info = f" prev_avg={prev_avg:.6f} curr_avg={curr_avg:.6f}"

        # Short status printed every epoch (diagnostic: show G loss, best, and whether it improved)
        print(f"Epoch {epoch} [no_improve: {no_improve}/{patience}] [effective_no_improve: {effective_no_improve}] [G: {g_loss_val:.6f}] [best: {best_g_loss:.6f}] [imp: {'Y' if improved else 'N'}]{win_info}")

        # Detailed logging and sample saving still follow the original rules
        if epoch % args.interval == 0 or improved:
            print(f"Epoch {epoch} [D loss: {d_loss_val:.6f}] [G loss: {g_loss_val:.6f}] [best: {best_g_loss:.6f}] [bonus_left: {bonus_remaining}]")
            try:
                gan.save_sample_images(args.save_dir, epoch)
            except Exception as e:
                print(f"샘플 저장 실패: {e}")

        # Early stopping: require minimum epochs and check effective no_improve
        if epoch >= min_epochs and effective_no_improve >= patience:
            print(f"⏱️ Early stopping triggered at epoch {epoch} (no improvement for {effective_no_improve} epochs, patience {patience}).")
            break

    final_model_path = f"{args.save_dir}/final_generator.h5"
    gan.generator.save(final_model_path)
    # If recon export requested, also save a run-local recon model
    export_reconstruction_model(gan.generator, X_train, os.path.join(args.save_dir, 'best_reconstructor.h5'), epochs=getattr(args, 'export_recon_epochs', 0), batch_size=args.batch_size)

    # Optional: build an image->image recon model (encoder + frozen generator) for inference.
    # This trains only the encoder for a few epochs to map images to the generator's latent space.
    def export_reconstruction_model(generator, train_data, save_path, epochs=0, batch_size=16):
        if epochs <= 0:
            return None
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        encoder_input = layers.Input(shape=img_shape)
        x = encoder_input
        x = layers.Conv2D(64, 4, strides=2, padding='same', activation='leaky_relu')(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same', activation='leaky_relu')(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same', activation='leaky_relu')(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same', activation='leaky_relu')(x)
        x = layers.Flatten()(x)
        latent = layers.Dense(args.latent_dim)(x)

        generator.trainable = False
        recon = generator(latent)
        recon_model = Model(encoder_input, recon, name='EncoderGeneratorRecon')
        recon_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='mae')
        recon_model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
        recon_model.save(save_path)
        return save_path

    # 평가 및 글로벌 베스트와 비교
    # 현재 지표: 이 실행에서의 best generator loss (작을수록 좋음)
    current_metric = best_g_loss if best_g_loss != float('inf') else float('inf')

    global_best_file = os.path.join('outputs', 'global_gan_best.json')
    os.makedirs('outputs', exist_ok=True)

    keep_run = True
    if os.path.exists(global_best_file):
        with open(global_best_file, 'r') as f:
            data = json.load(f)
        best_metric = data.get('best_metric', float('inf'))
        if current_metric < best_metric:
            # 더 좋음 -> 갱신
            shutil.copyfile(f"{args.save_dir}/best_generator.h5", os.path.join('outputs', 'global_best_generator.h5'))
            # export recon model if requested
            export_reconstruction_model(gan.generator, X_train, os.path.join('outputs', 'global_best_reconstructor.h5'), epochs=getattr(args, 'export_recon_epochs', 0), batch_size=args.batch_size)
            data = {
                'best_metric': current_metric,
                'saved_at': datetime.utcnow().isoformat(),
                'source_dir': args.save_dir
            }
            with open(global_best_file, 'w') as f:
                json.dump(data, f)
            print(f"✅ Run improved global best (metric {current_metric:.6f}). Global best updated.")
        else:
            # 좋지 않음 -> 삭제 옵션이 켜져 있으면 결과 제거
            if getattr(args, 'prune_if_worse', True):
                try:
                    shutil.rmtree(args.save_dir)
                    print(f"🗑️ Run did not beat global best (current {current_metric:.6f} >= best {best_metric:.6f}). Deleted {args.save_dir}.")
                    keep_run = False
                except Exception as e:
                    print(f"Failed to remove {args.save_dir}: {e}")
            else:
                print(f"Run did not beat global best (current {current_metric:.6f} >= best {best_metric:.6f}). Kept artifacts.")
    else:
        # 처음 베스트 설정
        shutil.copyfile(f"{args.save_dir}/best_generator.h5", os.path.join('outputs', 'global_best_generator.h5'))
        export_reconstruction_model(gan.generator, X_train, os.path.join('outputs', 'global_best_reconstructor.h5'), epochs=getattr(args, 'export_recon_epochs', 0), batch_size=args.batch_size)
        data = {
            'best_metric': current_metric,
            'saved_at': datetime.utcnow().isoformat(),
            'source_dir': args.save_dir
        }
        with open(global_best_file, 'w') as f:
            json.dump(data, f)
        print(f"✅ No previous global best. Saved current run as global best (metric {current_metric:.6f}).")

    if keep_run:
        print("\n✅ 학습 완료! 'best_generator.h5'와 'final_generator.h5'가 저장되었습니다.")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dcgan','anomaly'], default='dcgan', help='학습 모드 선택')
    parser.add_argument('--epochs', type=int, default=2000, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=0.0002, help='학습률 (Learning Rate)')
    parser.add_argument('--latent_dim', type=int, default=100, help='잠재 공간 차원')
    parser.add_argument('--interval', type=int, default=100, help='이미지 저장 및 로그 출력 간격')
    parser.add_argument('--save_dir', type=str, default=None, help='모델 저장 경로 (env SAVE_DIR 또는 config.get_save_dir)')
    parser.add_argument('--data_dir', type=str, default=None, help='학습 이미지 폴더 (config.get_data_paths train_dir 사용)')
    parser.add_argument('--img_size', type=int, default=128, help='정사각 이미지 크기')
    parser.add_argument('--channels', type=int, default=1, help='채널 수 (1=gray, 3=RGB)')
    parser.add_argument('--seed', type=int, default=None, help='재현성을 위한 시드')
    parser.add_argument('--max_images', type=int, default=None, help='학습에 사용할 최대 이미지 수 제한')
    # anomaly specific
    parser.add_argument('--class', dest='class_name', default=None, help='MVTec class name for anomaly mode (env CLASS_NAME fallback)')
    parser.add_argument('--mvtec_root', type=str, default=None, help='MVTec root folder (env MVTEC_ROOT fallback)')
    parser.add_argument('--dry_run', action='store_true', help='데이터 로드만 확인하고 종료')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum absolute improvement to count')
    parser.add_argument('--min_epochs', type=int, default=10, help='Minimum epochs before early stopping allowed')
    parser.add_argument('--stagnation_window', type=int, default=5, help='Window size for stagnation detection (epochs)')
    parser.add_argument('--max_improve_ratio', type=float, default=2.0, help='If avg(prev_window)/avg(curr_window) > this, treat as large improvement')
    parser.add_argument('--bonus_epochs_on_large_improve', type=int, default=3, help='Extra patience epochs after a large improvement')
    parser.add_argument('--prune_if_worse', type=bool, default=True, help='If True, delete run artifacts when not improving global best')
    parser.add_argument('--export_recon_epochs', type=int, default=0, help='If >0, train a lightweight encoder+frozen-generator recon model for this many epochs and save as best_reconstructor.h5 (and global_best_reconstructor.h5 when improved)')

    args = parser.parse_args()

    # Resolve shared paths via config
    if not args.class_name:
        args.class_name = config.DATA_CLASS
    if not args.mvtec_root:
        args.mvtec_root = str(config.DATA_ORIGIN)
    if not args.data_dir:
        train_dir, _ = config.get_data_paths(args.class_name)
        args.data_dir = str(train_dir)
    if not args.save_dir:
        args.save_dir = str(config.get_save_dir('saved_models'))

    if args.mode in ('dcgan', 'anomaly'):
        # For backward compatibility, call existing run_training for the DCGAN-style AnomalyGAN
        if args.mode == 'anomaly':
            # prefer MVTec layout if available
            imgs = collect_train_images(args.mvtec_root, args.class_name or find_first_class(args.mvtec_root), size=(args.img_size,args.img_size))
            if args.dry_run:
                print(f"Dry run: loaded {imgs.shape[0]} images for class {args.class_name}")
                return
            os.makedirs(args.save_dir, exist_ok=True)
            agan = AnomalyGAN((args.img_size,args.img_size,args.channels), args.latent_dim, args.lr)
            agan.train = None  # avoid attribute collision; use existing run_training
            # reuse run_training by constructing a minimal args namespace
            class SimpleArgs:
                pass
            ra = SimpleArgs()
            ra.epochs = args.epochs
            ra.batch_size = args.batch_size
            ra.lr = args.lr
            ra.latent_dim = args.latent_dim
            ra.interval = args.interval
            ra.save_dir = args.save_dir
            ra.data_dir = args.data_dir
            ra.img_size = args.img_size
            ra.channels = args.channels
            ra.seed = args.seed
            ra.max_images = args.max_images
            run_training(ra)
        else:
            run_training(args)

if __name__ == '__main__':
    train()