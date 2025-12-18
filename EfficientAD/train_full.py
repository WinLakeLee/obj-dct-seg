import os
import time
import argparse
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
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
                ae_out = F.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear', align_corners=False)

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

    # env/CLI resolution using shared config
    cli_train = args.train_dir
    cli_val = args.val_dir
    cli_save = args.save_dir

    train_dir, val_dir = config.get_data_paths()
    if cli_train:
        train_dir = cli_train
    if cli_val:
        val_dir = cli_val
    save_dir = config.get_save_dir(cli_save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare dataloaders
    train_loader = get_dataloader(str(train_dir), img_size=args.image_size, batch_size=args.batch_size)
    val_loader = get_dataloader(str(val_dir), img_size=args.image_size, batch_size=args.batch_size) if val_dir else None

    model = EfficientAD(image_size=args.image_size)
    model.student.to(device); model.ae.to(device)
    optimizer = model.optimizer

    best_val = float('inf')
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
        model.teacher_mean = torch.mean(outputs, dim=[0,2,3], keepdim=True)
        model.teacher_std = torch.std(outputs, dim=[0,2,3], keepdim=True)

    for epoch in range(1, args.epochs + 1):
        model.student.train(); model.ae.train()
        total_loss = 0.0
        batches = 0
        for imgs in train_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                teacher_out = model.teacher(imgs)
                teacher_out = model._normalize_teacher_output(teacher_out)

            student_out = model.student(imgs)
            ae_out = model.ae(imgs)
            if ae_out.shape[2:] != teacher_out.shape[2:]:
                ae_out = F.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear', align_corners=False)

            loss_st = model.calculated_hard_loss(teacher_out, student_out, q=0.99)
            loss_ae = F.mse_loss(ae_out, teacher_out)
            loss_st_ae = F.mse_loss(student_out, ae_out.detach())
            loss = loss_st + loss_ae + loss_st_ae

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_avg = total_loss / max(1, batches)
        print(f"Epoch {epoch}/{args.epochs} - train avg loss: {train_avg:.6f}")

        # validation
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            print(f"  Validation loss: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = save_checkpoint(str(save_dir), model, optimizer, epoch, best=True)
                print(f"  New best model saved: {best_path}")

        # periodic checkpoint
        if epoch % args.save_interval == 0:
            path = save_checkpoint(str(save_dir), model, optimizer, epoch, best=False)
            print(f"  Checkpoint saved: {path}")

    elapsed = time.time() - start_time
    print(f"Training finished. Time: {elapsed:.2f}s, best_val: {best_val}, best_path: {best_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', required=False, default=None)
    p.add_argument('--val-dir', required=False, default=None)
    p.add_argument('--save-dir', required=False, default="outputs/efficientad")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--save-interval', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train_loop(args)
