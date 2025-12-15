import os
import sys
import torch

# add project root to sys.path so imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PatchCore.patch_core import PatchCoreFromScratch


def make_loader(n=3):
    imgs = []
    for _ in range(n):
        imgs.append(torch.randn(1, 3, 224, 224))
    return imgs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, 'cuda_available:', torch.cuda.is_available())

    pc = PatchCoreFromScratch()
    # move backbone to device so forward runs on GPU if available
    try:
        pc.backbone.to(device)
    except Exception as e:
        print('Warning moving model to device:', e)

    train_loader = make_loader(3)
    train_loader = [img.to(device) for img in train_loader]

    pc.fit(train_loader)

    test_img = torch.randn(1, 3, 224, 224).to(device)
    score = pc.predict(test_img)
    print('anomaly_score:', score)


if __name__ == '__main__':
    main()
