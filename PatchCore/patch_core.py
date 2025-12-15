import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    # newer torchvision: use Weights enum
    from torchvision.models import Wide_ResNet50_2_Weights
    _WIDE_RESNET_WEIGHTS = Wide_ResNet50_2_Weights.DEFAULT
except Exception:
    _WIDE_RESNET_WEIGHTS = None
from sklearn.neighbors import NearestNeighbors
import numpy as np

logger = logging.getLogger(__name__)

class PatchCoreFromScratch:
    def __init__(self):
        # 1. 눈 (Backbone): 사전 학습된 Wide ResNet50 가져오기
        # prefer modern weights API if available to avoid deprecation warnings
        if _WIDE_RESNET_WEIGHTS is not None:
            self.backbone = models.wide_resnet50_2(weights=_WIDE_RESNET_WEIGHTS)
        else:
            # fallback for older torchvision versions
            self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.eval() # 학습 안 함 (평가 모드)
        
        # 특징을 낚아챌 레이어 지정 (Layer 2, Layer 3)
        self.features = []
        self.hooks = []
        self._register_hooks()
        
        self.memory_bank = [] # 기억 저장소

    def _hook_fn(self, module, input, output):
        # 레이어를 통과할 때 데이터를 가로채는 함수
        self.features.append(output)

    def _register_hooks(self):
        # ResNet의 layer2와 layer3 뒤에 도청 장치(Hook) 설치
        self.backbone.layer2.register_forward_hook(self._hook_fn)
        self.backbone.layer3.register_forward_hook(self._hook_fn)

    def extract_features(self, x):
        """
        이미지에서 Patch 단위의 특징 벡터들을 추출합니다.
        Input: (Batch_Size, 3, H, W)
        Output: (Total_Pixels, Feature_Dimension) -> KNN에 들어갈 벡터들
        """
        self.features = [] # Hook으로 채워질 리스트 초기화
        with torch.no_grad():
            self.backbone(x)
        
        # 1. 특징 맵 가져오기 (Hook으로 낚아챈 것들)
        # ResNet50 기준:
        # features[0] (Layer2): [Batch, 512, 28, 28] -> 큼 (디테일)
        # features[1] (Layer3): [Batch, 1024, 14, 14] -> 작음 (전체적 의미)
        f2 = self.features[0] 
        f3 = self.features[1] 

        # 2. 크기 맞추기 (Upsampling)
        # 작은 Layer3(f3)를 Layer2(f2)의 크기(28x28)로 강제로 늘립니다.
        # mode='bilinear': 부드럽게 늘리기
        f3_resized = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=True)

        # 3. 합치기 (Concatenation)
        # 채널(dim=1) 방향으로 합칩니다.
        # 결과: [Batch, 512 + 1024, 28, 28] = [Batch, 1536, 28, 28]
        concat_features = torch.cat([f2, f3_resized], dim=1)

        # 4. Patching (지역 정보 집계) - ⭐ 여기가 핵심!
        # 픽셀 하나의 값만 쓰는 게 아니라, 3x3 주변 정보를 평균 내서 씁니다.
        # 이렇게 하면 픽셀이 살짝 밀려도 비슷하게 인식합니다 (Robustness).
        # AvgPool2d(kernel_size=3, stride=1, padding=1) -> 크기는 유지됨
        patch_features = F.avg_pool2d(concat_features, kernel_size=3, stride=1, padding=1)

        # 5. 모양 변경 (Flatten)
        # KNN은 (N, Dimension) 형태의 2차원 표만 이해합니다.
        # [Batch, Channel, H, W] -> [Batch, H, W, Channel]
        patch_features = patch_features.permute(0, 2, 3, 1)
        
        # [Batch * H * W, Channel] 형태로 쫙 폅니다.
        # 예: 이미지 1장(28x28=784픽셀) -> (784, 1536) 크기의 벡터 뭉치
        output_features = patch_features.reshape(-1, patch_features.shape[-1])

        return output_features.cpu() # 메모리 절약을 위해 CPU로 보냄
    
    def fit(self, train_loader, checkpoint_dir=None, checkpoint_interval=100, n_neighbors=9):
        """Build memory bank from training loader.

        Args:
            train_loader: iterable yielding batched image tensors
            checkpoint_dir: optional directory to save partial/final checkpoints
            checkpoint_interval: save partial features every N batches
        """
        logger.info("🧠 정상 패턴 기억 중...")
        features_list = []
        batch_count = 0

        # store checkpoint params on the instance for other code if needed
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_interval = checkpoint_interval

        for imgs in train_loader:
            batch_count += 1
            feats = self.extract_features(imgs)
            feats_np = feats.cpu().numpy()
            features_list.append(feats_np)

            if checkpoint_dir and checkpoint_interval and (batch_count % checkpoint_interval == 0):
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    partial = np.concatenate(features_list)
                    path = os.path.join(checkpoint_dir, f'partial_features_until_batch_{batch_count}.npy')
                    np.save(path, partial)
                    logger.info(f'Checkpoint: saved partial features to {path}')
                except Exception as e:
                    logger.warning(f'Failed to save checkpoint at batch {batch_count}: {e}')

        # concat all features
        if len(features_list) == 0:
            self.memory_bank = np.zeros((0, 0))
        else:
            self.memory_bank = np.concatenate(features_list)

        # fit knn
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(self.memory_bank)

        # final save
        if checkpoint_dir:
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                mb_path = os.path.join(checkpoint_dir, 'memory_bank.npy')
                np.save(mb_path, self.memory_bank)
                logger.info(f'Saved final memory_bank to {mb_path}')
                try:
                    import joblib
                    knn_path = os.path.join(checkpoint_dir, 'knn.pkl')
                    joblib.dump(self.knn, knn_path)
                    logger.info(f'Saved final KNN to {knn_path}')
                except Exception:
                    logger.warning('joblib not available, skipping knn save')
            except Exception as e:
                logger.warning(f'Failed to save final checkpoint: {e}')

    def predict(self, img, kneighbors_batch=4096):
        """Compute anomaly score for `img`.

        Accepts a single image tensor (3,H,W) or a batch tensor (B,3,H,W).
        Returns a single score for single input or list of scores for batch.
        `kneighbors_batch` controls how many query patches are passed to KNN at once.
        """
        if not hasattr(self, 'knn'):
            raise RuntimeError('KNN not fitted. Call fit() first.')

        single = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            single = True

        # extract features for whole batch
        test_feat_t = self.extract_features(img)
        test_feat = test_feat_t.cpu().numpy()

        total = test_feat.shape[0]
        B = img.shape[0]
        if B == 0 or total == 0:
            return [] if not single else 0.0

        patches_per_img = total // B

        # query KNN in chunks to save memory
        distances_chunks = []
        for start in range(0, total, kneighbors_batch):
            end = min(total, start + kneighbors_batch)
            dists, _ = self.knn.kneighbors(test_feat[start:end])
            distances_chunks.append(dists)

        distances = np.vstack(distances_chunks)

        scores = []
        for i in range(B):
            s = distances[i * patches_per_img:(i + 1) * patches_per_img].mean()
            scores.append(float(s))

        return scores[0] if single else scores