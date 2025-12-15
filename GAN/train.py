import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import os
import json
import shutil
from datetime import datetime
import argparse
from .gan import FastAnomalyGAN, AnomalyGAN
import glob
from pathlib import Path
from PIL import Image


def find_first_class(root="data/mvtec"):
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"MVTec root not found: {root}")
    for c in sorted(p.iterdir()):
        if c.is_dir():
            return c.name
    raise RuntimeError("No class folders found under data/mvtec")


def collect_train_images(root, class_name, size=(128, 128)):
    base = Path(root) / class_name / 'train'
    patterns = [str(base / '**' / '*.png'), str(base / '**' / '*.jpg'), str(base / '**' / '*.jpeg')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No training images found for class {class_name} under {base}")

    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB')
        img = img.resize(size, Image.BICUBIC)
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.mean(arr, axis=2, keepdims=True)
        arr = arr / 127.5 - 1.0
        imgs.append(arr)
    return np.stack(imgs, axis=0)
import matplotlib.pyplot as plt
from PIL import Image
import glob

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
        X_train = np.random.normal(0, 1, (1000, img_shape[0], img_shape[1], img_shape[2]))
        return X_train

    # 가능한 확장자 수집
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(os.path.join(data_dir, e))))

    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    if len(paths) == 0:
        print(f"⚠️ 경고: '{data_dir}'에서 이미지를 찾을 수 없습니다. 랜덤 데이터를 사용합니다.")
        X_train = np.random.normal(0, 1, (1000, img_shape[0], img_shape[1], img_shape[2]))
        return X_train

    imgs = []
    for p in paths:
        try:
            im = Image.open(p)
            # 채널 변환
            if img_shape[-1] == 1:
                im = im.convert('L')
            else:
                im = im.convert('RGB')
            im = im.resize((img_shape[1], img_shape[0]), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
            if img_shape[-1] == 1:
                arr = arr[:, :, None]
            # 스케일을 -1 ~ 1로 변환 (입력 범위가 0~255일 때)
            if arr.max() > 1.0:
                arr = (arr / 127.5) - 1.0
            imgs.append(arr)
        except Exception as e:
            print(f"이미지 로드 실패: {p} -> {e}")

    X_train = np.stack(imgs, axis=0)
    return X_train

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
        if g_loss + min_delta < best_g_loss:
            best_g_loss = float(g_loss)
            gan.generator.save(f"{args.save_dir}/best_generator.h5")
            print(f"Epoch {epoch}: 🔥 새로운 Best Model 저장됨! (G Loss: {g_loss:.6f})")
            improved = True
            no_improve = 0
            bonus_remaining = bonus_epochs
        else:
            # If we haven't reached minimum epochs, don't count as failure
            if epoch < min_epochs:
                no_improve = 0
            else:
                # If we have at least two windows, perform stagnation detection
                if len(g_loss_history) >= 2 * stag_w:
                    prev_avg = float(np.mean(g_loss_history[-2 * stag_w:-stag_w]))
                    curr_avg = float(np.mean(g_loss_history[-stag_w:]))
                    delta = prev_avg - curr_avg
                    if delta < min_delta:
                        no_improve += 1
                    else:
                        no_improve = 0

                    # Large improvement handling: grant bonus epochs for further training
                    if prev_avg / max(curr_avg, 1e-12) > max_ratio:
                        print(f"Epoch {epoch}: Large improvement detected (ratio {prev_avg/curr_avg:.2f}), granting bonus {bonus_epochs} epochs")
                        bonus_remaining = max(bonus_remaining, bonus_epochs)
                        no_improve = 0
                else:
                    no_improve += 1

        # Consume bonus if present
        if bonus_remaining > 0:
            effective_no_improve = 0
            bonus_remaining -= 1
        else:
            effective_no_improve = no_improve

        if epoch % args.interval == 0 or improved:
            print(f"Epoch {epoch} [D loss: {d_loss[0]:.6f}] [G loss: {g_loss:.6f}] [best: {best_g_loss:.6f}] [no_improve: {no_improve}/{patience}] [bonus_left: {bonus_remaining}]")
            gan.save_sample_images(epoch)

        # Early stopping: require minimum epochs and check effective no_improve
        if epoch >= min_epochs and effective_no_improve >= patience:
            print(f"⏱️ Early stopping triggered at epoch {epoch} (no improvement for {effective_no_improve} epochs, patience {patience}).")
            break

    final_model_path = f"{args.save_dir}/final_generator.h5"
    gan.generator.save(final_model_path)

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
    parser.add_argument('--mode', choices=['dcgan','anomaly','fast'], default='dcgan', help='학습 모드 선택')
    parser.add_argument('--epochs', type=int, default=2000, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=0.0002, help='학습률 (Learning Rate)')
    parser.add_argument('--latent_dim', type=int, default=100, help='잠재 공간 차원')
    parser.add_argument('--interval', type=int, default=100, help='이미지 저장 및 로그 출력 간격')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='모델 저장 경로')
    parser.add_argument('--data_dir', type=str, default='data/archive/train_images', help='학습 이미지 폴더 (PatchCore와 동일한 구조 사용)')
    parser.add_argument('--img_size', type=int, default=128, help='정사각 이미지 크기')
    parser.add_argument('--channels', type=int, default=1, help='채널 수 (1=gray, 3=RGB)')
    parser.add_argument('--seed', type=int, default=None, help='재현성을 위한 시드')
    parser.add_argument('--max_images', type=int, default=None, help='학습에 사용할 최대 이미지 수 제한')
    # anomaly/fast specific
    parser.add_argument('--class', dest='class_name', default=None, help='MVTec class name for anomaly mode')
    parser.add_argument('--mvtec_root', type=str, default='data/mvtec', help='MVTec root folder')
    parser.add_argument('--dry_run', action='store_true', help='데이터 로드만 확인하고 종료')
    parser.add_argument('--pretrained', type=str, default=None, help='학습된 AnomalyGAN generator weights (fast mode)')
    parser.add_argument('--fine_tune_epochs', type=int, default=5, help='FastAnomalyGAN fine-tune epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum absolute improvement to count')
    parser.add_argument('--min_epochs', type=int, default=10, help='Minimum epochs before early stopping allowed')
    parser.add_argument('--stagnation_window', type=int, default=5, help='Window size for stagnation detection (epochs)')
    parser.add_argument('--max_improve_ratio', type=float, default=2.0, help='If avg(prev_window)/avg(curr_window) > this, treat as large improvement')
    parser.add_argument('--bonus_epochs_on_large_improve', type=int, default=3, help='Extra patience epochs after a large improvement')
    parser.add_argument('--prune_if_worse', type=bool, default=True, help='If True, delete run artifacts when not improving global best')

    args = parser.parse_args()

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
    elif args.mode == 'fast':
        # Fast mode: require pretrained weights
        if args.dry_run:
            # just verify MVTec dataset
            imgs = collect_train_images(args.mvtec_root, args.class_name or find_first_class(args.mvtec_root), size=(args.img_size,args.img_size))
            print(f"Dry run: loaded {imgs.shape[0]} images for class {args.class_name}")
            return
        if not args.pretrained:
            raise RuntimeError('Fast mode requires --pretrained path to AnomalyGAN generator weights')
        imgs = collect_train_images(args.mvtec_root, args.class_name or find_first_class(args.mvtec_root), size=(args.img_size,args.img_size))
        os.makedirs(args.save_dir, exist_ok=True)
        agan = AnomalyGAN((args.img_size,args.img_size,args.channels), args.latent_dim, args.lr)
        agan.generator.load_weights(args.pretrained)
        fgan = FastAnomalyGAN(img_shape=(args.img_size,args.img_size,args.channels))
        fgan.copy_decoder_from_anomaly_generator(agan.generator)
        if args.fine_tune_epochs > 0:
            fgan.train(imgs, epochs=args.fine_tune_epochs, batch_size=args.batch_size, save_dir=args.save_dir)
        fgen_w_path = os.path.join(args.save_dir, 'fgan_final_gen.h5')
        fgan.generator.save_weights(fgen_w_path)
        print(f'FastAnomalyGAN weights saved to {fgen_w_path}')


if __name__ == '__main__':
    train()