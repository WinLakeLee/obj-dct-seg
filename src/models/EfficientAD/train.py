import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Ensure project root is on sys.path so `configs` package can be imported
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from PDN import EfficientAD, get_dataloader


def save_checkpoint(save_dir, model, optimizer, epoch, best=False):
    os.makedirs(save_dir, exist_ok=True)
    prefix = "best" if best else f"epoch{epoch}"
    path = os.path.join(save_dir, f"{prefix}_checkpoint.pth")
    state = {
        "epoch": epoch,
        "student_state": model.student.state_dict(),
        "ae_state": model.ae.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)
    return path


def validate(model, dataloader, device):
    model.student.eval()
    model.ae.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            teacher_out = model.teacher(imgs)
            teacher_out = model._normalize_teacher_output(teacher_out)
            student_out = model.student(imgs)
            ae_out = model.ae(imgs)
            if ae_out.shape[2:] != teacher_out.shape[2:]:
                ae_out = F.interpolate(
                    ae_out,
                    size=teacher_out.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_st = model.calculated_hard_loss(teacher_out, student_out, q=0.99)
            loss_ae = F.mse_loss(ae_out, teacher_out)
            loss_st_ae = F.mse_loss(student_out, ae_out)
            loss = loss_st + loss_ae + loss_st_ae
            total += loss.item() * imgs.shape[0]
            count += imgs.shape[0]
    return total / max(1, count)


def train_loop(args):
    # Load .env if present so MVTEC_ROOT / TRAIN_DIR / VAL_DIR / SAVE_DIR can be configured centrally
    load_dotenv()

    # env/CLI resolution using shared config (align with GAN)
    if args.data_origin:
        config.DATA_ORIGIN = Path(args.data_origin)
    cls = args.class_name or config.DATA_CLASS

    cli_train = args.train_dir
    cli_val = args.val_dir
    cli_save = args.save_dir

    train_dir_default, val_dir_default = config.get_data_paths(cls)
    train_dir = Path(cli_train) if cli_train else train_dir_default
    val_dir = Path(cli_val) if cli_val else val_dir_default
    save_dir = config.get_save_dir(cli_save)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare dataloaders
    train_loader = get_dataloader(
        str(train_dir), img_size=args.image_size, batch_size=args.batch_size
    )
    val_loader = (
        get_dataloader(
            str(val_dir), img_size=args.image_size, batch_size=args.batch_size
        )
        if val_dir
        else None
    )

    model = EfficientAD(image_size=args.image_size)
    model.student.to(device)
    model.ae.to(device)
    optimizer = model.optimizer

    best_val = float("inf")
    best_path = None
    start_time = time.time()

    # Precompute teacher stats on train subset (same as original PDN.train)
    with torch.no_grad():
        outputs = []
        for i, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            outputs.append(model.teacher(imgs))
            if i >= 10:  # limit number of batches used for statistics for speed
                break
        outputs = torch.cat(outputs, dim=0)
        model.teacher_mean = torch.mean(outputs, dim=[0, 2, 3], keepdim=True)
        model.teacher_std = torch.std(outputs, dim=[0, 2, 3], keepdim=True)

    # Early-stopping / stagnation parameters (GAN-style)
    no_improve = 0
    val_history = []
    bonus_remaining = 0
    min_epochs = getattr(args, "min_epochs", 10)
    stag_w = getattr(args, "stagnation_window", 5)
    max_ratio = getattr(args, "max_improve_ratio", 2.0)
    bonus_epochs = getattr(args, "bonus_epochs_on_large_improve", 3)
    for epoch in range(1, args.max_epochs + 1):
        model.student.train()
        model.ae.train()
    for epoch in range(1, args.max_epochs + 1):
        model.student.train()
        model.ae.train()
        total_loss = 0.0
        batches = 0
        for epoch in range(1, args.max_epochs + 1):
            teacher_out = model.teacher(imgs)
            teacher_out = model._normalize_teacher_output(teacher_out)

            student_out = model.student(imgs)
            ae_out = model.ae(imgs)
            if ae_out.shape[2:] != teacher_out.shape[2:]:
                ae_out = F.interpolate(
                    ae_out,
                    size=teacher_out.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_st = model.calculated_hard_loss(teacher_out, student_out, q=0.99)
            loss_ae = F.mse_loss(ae_out, teacher_out)
            loss_st_ae = F.mse_loss(student_out, ae_out.detach())
            loss = loss_st + loss_ae + loss_st_ae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_avg = total_loss / max(1, batches)
        print(f"Epoch {epoch}/{args.max_epochs} - train avg loss: {train_avg:.6f}")

        # validation
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            print(f"  Validation loss: {val_loss:.6f}")
            min_delta = float(getattr(args, "min_delta", 1e-4))
            val_history.append(float(val_loss))

            improved = False
            # Direct improvement check (absolute improvement)
            if float(val_loss) < best_val - min_delta:
                best_val = float(val_loss)
                best_path = save_checkpoint(
                    str(save_dir), model, optimizer, epoch, best=True
                )
                print(f"  New best model saved: {best_path}")
                no_improve = 0
                improved = True
            else:
                # If we haven't reached minimum epochs, don't count as failure
                if epoch < min_epochs:
                    no_improve = 0
                else:
                    # Sliding-window stagnation detection
                    if len(val_history) >= 2 * stag_w and stag_w > 0:
                        prev_avg = float(np.mean(val_history[-2 * stag_w : -stag_w]))
                        curr_avg = float(np.mean(val_history[-stag_w:]))
                        if curr_avg + min_delta < prev_avg:
                            # improvement detected in sliding window
                            no_improve = 0
                            improved = True
                            # Large improvement handling: grant bonus epochs
                            if prev_avg / max(curr_avg, 1e-12) > max_ratio:
                                print(
                                    f"Epoch {epoch}: Large improvement detected (ratio {prev_avg/curr_avg:.2f}), granting bonus {bonus_epochs} epochs"
                                )
                                bonus_remaining = max(bonus_remaining, bonus_epochs)
                        else:
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

            # Diagnostic sliding-window info
            win_info = ""
            if stag_w > 0 and len(val_history) >= 2 * stag_w:
                prev_avg = float(np.mean(val_history[-2 * stag_w : -stag_w]))
                curr_avg = float(np.mean(val_history[-stag_w:]))
                win_info = f" prev_avg={prev_avg:.6f} curr_avg={curr_avg:.6f}"
                print(f"  Window info:{win_info}")

            if epoch >= min_epochs and effective_no_improve >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch} (no improvement for {effective_no_improve} evals, patience {args.patience})."
                )
                break
            prev_avg = float(np.mean(val_history[-2 * stag_w : -stag_w]))
            curr_avg = float(np.mean(val_history[-stag_w:]))
            win_info = f" prev_avg={prev_avg:.6f} curr_avg={curr_avg:.6f}"
            print(f"  Window info:{win_info}")

            if epoch >= min_epochs and effective_no_improve >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch} (no improvement for {effective_no_improve} evals, patience {args.patience})."
                )
    # Persist run metrics for sweep selection
    best_ckpt = (
        Path(best_path)
        if best_path
        else Path(save_dir) / f"epoch{epoch}_checkpoint.pth"
    )
    metrics = {
        "best_val": float(best_val) if best_val is not None else None,
        "best_checkpoint": str(best_ckpt),
        "max_epochs": args.max_epochs,
        "stopped_epoch": epoch,
        "patience": args.patience,
        "min_delta": float(getattr(args, "min_delta", 1e-4)),
        "min_epochs": int(getattr(args, "min_epochs", 10)),
        "stagnation_window": int(getattr(args, "stagnation_window", 5)),
        "max_improve_ratio": float(getattr(args, "max_improve_ratio", 2.0)),
        "bonus_epochs_on_large_improve": int(
            getattr(args, "bonus_epochs_on_large_improve", 3)
        ),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
    }
    metrics_path = Path(save_dir) / "metrics.json"
    metrics_txt = Path(save_dir) / "metrics.txt"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    metrics_txt.write_text(
        "\n".join(
            [
                f"best_val: {metrics['best_val']}",
                f"best_checkpoint: {metrics['best_checkpoint']}",
                f"max_epochs: {metrics['max_epochs']}",
                f"stopped_epoch: {metrics['stopped_epoch']}",
                f"patience: {metrics['patience']}",
                f"min_delta: {metrics['min_delta']}",
                f"batch_size: {metrics['batch_size']}",
                f"image_size: {metrics['image_size']}",
            ]
        )
    )

    # --- Post-training: verify scratch detection on instance_segmentation images ---
    try:
        inst_root = Path("data/instance_segmentation")
        if inst_root.exists():
            # prefer test, then valid, then train
            for sub in ("test/images", "valid/images", "train/images"):
                verify_dir = inst_root / sub
                if verify_dir.exists():
                    print(f"\n🔎 Post-training verification on: {verify_dir}")
                    # load detect_anomaly_pipeline dynamically to avoid import issues
                    from importlib.machinery import SourceFileLoader

                    dap_path = (
                        Path(__file__).resolve().parents[2]
                        / "yolo_training"
                        / "detect_anomaly_pipeline.py"
                    )
                    if not dap_path.exists():
                        dap_path = (
                            Path(__file__).resolve().parents[3]
                            / "src"
                            / "models"
                            / "yolo_training"
                            / "detect_anomaly_pipeline.py"
                        )
                    if dap_path.exists():
                        dap_mod = SourceFileLoader(
                            "detect_anomaly_pipeline", str(dap_path)
                        ).load_module()
                        Pipeline = getattr(dap_mod, "ScratchDetectionPipeline")
                        # choose checkpoint: prefer best_path, else last epoch checkpoint in save_dir
                        ckpt = best_path or (
                            Path(save_dir) / f"epoch{epoch}_checkpoint.pth"
                        )
                        try:
                            pipeline = Pipeline(
                                yolo_model_path=None,
                                patchcore_checkpoint=None,
                                device="cuda",
                                conf_threshold=0.25,
                                anomaly_threshold=0.0,
                                anomaly_model="efficientad",
                                gan_generator_path=None,
                                efficientad_checkpoint=str(ckpt) if ckpt else None,
                                efficientad_image_size=args.image_size,
                            )
                            out_dir = Path(save_dir) / "verify_results"
                            pipeline.process_directory(
                                str(verify_dir), save_dir=str(out_dir)
                            )
                        except Exception as e:
                            print(f"Post-verification failed: {e}")
                    else:
                        print(
                            "detect_anomaly_pipeline.py를 찾을 수 없어 검증을 건너뜁니다."
                        )
                    break
        else:
            print(
                "data/instance_segmentation 폴더가 없어 post-training 검증을 건너뜁니다."
            )
    except Exception as e:
        print(f"Post-training verification encountered an error: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-dir",
        required=False,
        default=None,
        help="Training directory (overrides class-based resolution)",
    )
    p.add_argument(
        "--val-dir",
        required=False,
        default=None,
        help="Validation directory (overrides class-based resolution)",
    )
    p.add_argument(
        "--save-dir",
        required=False,
        default="outputs/efficientad",
        help="Checkpoint directory (defaults to config.get_save_dir)",
    )
    p.add_argument(
        "--class-name",
        required=False,
        default=None,
        help="Target class name (defaults to env CLASS_NAME)",
    )
    p.add_argument(
        "--data-origin",
        dest="data_origin",
        required=False,
        default=None,
        help="Override data/mvtec root (GAN/PatchCore alignment)",
    )
    p.add_argument(
        "--mvtec-root",
        dest="data_origin",
        required=False,
        default=None,
        help="(deprecated) use --data-origin",
    )
    p.add_argument(
        "--epochs",
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=1000,
        help="Upper bound on epochs; early stopping can cut sooner",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop if no validation improvement for this many evals",
    )
    p.add_argument(
        "--min-delta",
        dest="min_delta",
        type=float,
        default=1e-4,
        help="Minimum absolute improvement in validation loss to count as improvement (set 0 to disable)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--min-epochs",
        dest="min_epochs",
        type=int,
        default=10,
        help="Minimum number of epochs before early stopping can occur",
    )
    p.add_argument(
        "--stagnation-window",
        dest="stagnation_window",
        type=int,
        default=5,
        help="Sliding-window size for stagnation detection",
    )
    p.add_argument(
        "--max-improve-ratio",
        dest="max_improve_ratio",
        type=float,
        default=2.0,
        help="Ratio threshold to consider an improvement as large (grants bonus epochs)",
    )
    p.add_argument(
        "--bonus-epochs-on-large-improve",
        dest="bonus_epochs_on_large_improve",
        type=int,
        default=3,
        help="Number of extra epochs granted when a large improvement is detected",
    )
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_loop(args)
