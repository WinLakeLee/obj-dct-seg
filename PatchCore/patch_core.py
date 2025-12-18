import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
import numpy as np
from torchvision import transforms
import json
from pathlib import Path

# ---------------------------------------------------------
# 1. 성능 최적화: TF32 활성화 (Ampere GPU 이상에서 속도 향상)
# ---------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# torchvision 최신 버전 호환
try:
    from torchvision.models import Wide_ResNet50_2_Weights

    _WIDE_RESNET_WEIGHTS = Wide_ResNet50_2_Weights.DEFAULT
except Exception:
    _WIDE_RESNET_WEIGHTS = None

# FAISS 유무 확인
try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Triton availability (required for torch.compile -> inductor backend)
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

logger = logging.getLogger(__name__)


class PatchCoreOptimized:
    def __init__(
        self, backbone_name="wide_resnet50_2", sampling_ratio=0.01, use_fp16=True
    ):
        """
        Args:
            sampling_ratio (float): 메모리 뱅크 샘플링 비율.
            use_fp16 (bool): True일 경우 FP16(반정밀도) 모드를 사용하여 속도 향상 및 메모리 절약.
        """
        self.sampling_ratio = sampling_ratio
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Backbone 로드
        if backbone_name == "wide_resnet50_2":
            if _WIDE_RESNET_WEIGHTS is not None:
                self.backbone = models.wide_resnet50_2(weights=_WIDE_RESNET_WEIGHTS)
            else:
                self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)

        self.backbone.eval()
        self.backbone.to(self.device)

        # ---------------------------------------------------------
        # 2. 성능 최적화: FP16 모드 (메모리 절반, 속도 증가)
        # ---------------------------------------------------------
        if self.use_fp16:
            self.backbone.half()
            logger.info("🚀 FP16(Half Precision) 모드가 활성화되었습니다.")

        # ---------------------------------------------------------
        # 3. 성능 최적화: torch.compile (PyTorch 2.x 이상, triton 필요)
        # ---------------------------------------------------------
        if hasattr(torch, "compile") and HAS_TRITON:
            try:
                self.backbone = torch.compile(self.backbone)
                logger.info("🚀 PyTorch 2.0 Compilation이 적용되었습니다.")
            except Exception as e:
                logger.warning(f"Compilation 실패 (무시 가능): {e}")
        else:
            logger.info("torch.compile 건너뜀 (triton 없음 또는 환경 미지원)")

        # 특징 추출을 위한 Hook 설정
        self.features = []
        self._register_hooks()

        self.memory_bank = None
        self.knn = None
        self.faiss_index = None
        self.n_neighbors = 9

    def to(self, device):
        """Move backbone and update internal device tracking."""
        self.device = torch.device(device)
        self.backbone.to(self.device)
        return self

    def _hook_fn(self, module, input, output):
        # FP16 모드일 경우 Hook 출력도 FP16일 수 있으므로 필요시 처리 가능
        self.features.append(output)

    def _register_hooks(self):
        self.backbone.layer2.register_forward_hook(self._hook_fn)
        self.backbone.layer3.register_forward_hook(self._hook_fn)

    def extract_features(self, x):
        """이미지 배치를 입력받아 (N_patches, Dim) 형태의 특징 벡터 반환"""
        self.features = []

        # 입력 데이터 장치 및 타입 변환
        x = x.to(self.device)
        if self.use_fp16:
            x = x.half()

        with torch.no_grad():
            self.backbone(x)

        # Feature Map 가져오기
        f2 = self.features[0]
        f3 = self.features[1]

        # Upsampling & Concatenation
        # F.interpolate는 FP16에서 동작하지만, 안정성을 위해 float32로 변환해서 계산하는 경우도 있음.
        # 여기서는 속도를 위해 그대로 진행하되 align_corners=True는 유지
        f3_resized = F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=True
        )
        concat_features = torch.cat([f2, f3_resized], dim=1)

        # Average Pooling (Smoothing)
        patch_features = F.avg_pool2d(
            concat_features, kernel_size=3, stride=1, padding=1
        )

        # (Batch, C, H, W) -> (Batch, H, W, C) -> (N, C)
        patch_features = patch_features.permute(0, 2, 3, 1)
        output_features = patch_features.reshape(-1, patch_features.shape[-1])

        # 주의: Faiss(CPU)나 Sklearn은 float32만 받습니다.
        # 따라서 반환 시에는 float32로 캐스팅하여 CPU로 보냅니다.
        return output_features.float().cpu()

    @staticmethod
    def get_train_transforms(
        resize_size=256,
        crop_size=224,
        random_crop=False,
        hflip=False,
        rotation=0.0,
        color_jitter=0.0,
    ):
        """PatchCore 학습용 데이터 증강 파이프라인."""
        tfs = [transforms.Resize(resize_size)]
        if random_crop:
            tfs.append(transforms.RandomCrop(crop_size))
        else:
            tfs.append(transforms.CenterCrop(crop_size))
        if hflip:
            tfs.append(transforms.RandomHorizontalFlip(p=0.5))
        if rotation and rotation > 0:
            tfs.append(transforms.RandomRotation(degrees=rotation))
        if color_jitter and color_jitter > 0:
            tfs.append(
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter / 2,
                )
            )
        tfs.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transforms.Compose(tfs)

    def predict_tta(self, img, score_type="max", kneighbors_batch=None):
        """Test-Time Augmentation 기반 평균 점수 계산."""
        augmented_imgs = [
            img,
            transforms.functional.hflip(img),
            transforms.functional.rotate(img, angle=5),
            transforms.functional.rotate(img, angle=-5),
        ]

        scores_list = []
        for aug_img in augmented_imgs:
            score = self.predict(
                aug_img, score_type=score_type, kneighbors_batch=kneighbors_batch
            )
            scores_list.append(score[0])

        final_score = np.mean(scores_list)
        return [final_score]

    def _compute_greedy_coreset_indices(
        self, features: np.ndarray, sampling_ratio: float
    ) -> np.ndarray:
        """
        PatchCore의 핵심: K-Center Greedy 알고리즘
        무작위가 아니라, 가장 유의미한(거리가 먼) 특징들을 골라냅니다.
        """
        sample_size = int(features.shape[0] * sampling_ratio)
        if sample_size >= features.shape[0]:
            return np.arange(features.shape[0])

        logger.info(
            f"🧠 Coreset Sampling 시작: {features.shape[0]} -> {sample_size} (정확도 향상 중...)"
        )

        # 1. 속도를 위해 Random Projection으로 차원 축소 (예: 1024 -> 128)
        # 차원이 줄어도 점들 간의 거리 비율은 유지된다는 존슨-린덴슈트라우스 보조정리 활용
        reducer = SparseRandomProjection(n_components="auto", eps=0.9)
        reduced_features = reducer.fit_transform(features)

        # 2. Greedy Selection
        # 첫 번째 점은 무작위 선택
        selector = [np.random.randint(features.shape[0])]
        selected_indices = [selector[0]]

        # 가장 가까운 중심점까지의 거리 저장
        # 초기에는 첫 번째 선택된 점과의 거리로 초기화
        dist_matrix = np.linalg.norm(
            reduced_features - reduced_features[selector[0]], axis=1
        )

        for _ in range(1, sample_size):
            # 현재 선택된 점들로부터 가장 '멀리' 있는 점을 다음 점으로 선택
            # (가장 잘 대변되지 않은 영역을 커버하기 위해)
            next_index = np.argmax(dist_matrix)

            # 선택된 점 추가
            selected_indices.append(next_index)

            # 거리 갱신: 기존 거리 vs 새로 선택된 점과의 거리 중 더 작은 값 유지
            new_dist = np.linalg.norm(
                reduced_features - reduced_features[next_index], axis=1
            )
            dist_matrix = np.minimum(dist_matrix, new_dist)

        return np.array(selected_indices)

    def fit(
        self,
        train_loader,
        n_neighbors=9,
        checkpoint_dir=None,
        checkpoint_interval=None,
    ):
        logger.info("🧠 학습 시작: 특징 추출 및 메모리 뱅크 구축...")
        features_list = []

        # 배치 단위 추출 (메모리 관리)
        for step, imgs in enumerate(train_loader, start=1):
            feats = self.extract_features(imgs)
            features_list.append(feats.numpy())

            if checkpoint_interval and step % checkpoint_interval == 0:
                logger.info("Processed %d batches so far", step)

        # 1. 전체 특징 합치기
        full_bank = np.concatenate(features_list, axis=0)

        # ---------------------------------------------------------
        # [수정] 2. 성능 최적화: Random -> Coreset Sampling 변경
        # ---------------------------------------------------------
        if self.sampling_ratio < 1.0:
            indices = self._compute_greedy_coreset_indices(
                full_bank, self.sampling_ratio
            )
            self.memory_bank = full_bank[indices]
        else:
            self.memory_bank = full_bank

        self.memory_bank = np.ascontiguousarray(self.memory_bank.astype(np.float32))
        self.n_neighbors = n_neighbors
        self._build_index()

        if checkpoint_dir:
            self._save_checkpoint(checkpoint_dir)

    def _save_checkpoint(self, checkpoint_dir):
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        mb_path = ckpt_dir / "memory_bank.npy"
        np.save(str(mb_path), self.memory_bank)

        meta = {
            "sampling_ratio": self.sampling_ratio,
            "n_neighbors": self.n_neighbors,
            "use_fp16": self.use_fp16,
            "faiss": self.faiss_index is not None,
        }
        with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.knn is not None:
            try:
                import joblib

                joblib.dump(self.knn, str(ckpt_dir / "knn.pkl"))
            except Exception as e:
                logger.warning("KNN 저장 실패 (무시 가능): %s", e)

    def _build_index(self):
        """KNN 또는 Faiss 인덱스 빌드"""
        dim = self.memory_bank.shape[1]

        if HAS_FAISS:
            # ---------------------------------------------------------
            # 4. 성능 최적화: FAISS IndexFactory 사용 (자동 최적화)
            # ---------------------------------------------------------
            # 데이터가 매우 많다면 'IVF1024,Flat' 등을 사용하여 근사 검색(속도↑) 가능
            # 여기서는 정확도를 위해 FlatL2를 쓰되 GPU 자원을 활용
            index_str = "Flat"

            try:
                # GPU 리소스 사용 시도
                res = faiss.StandardGpuResources()
                # 인덱스 생성
                index = faiss.index_factory(dim, index_str, faiss.METRIC_L2)

                # GPU로 이동 (메모리가 허용하는 경우)
                if torch.cuda.is_available():
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("🚀 FAISS: GPU 인덱싱 성공")

                index.add(self.memory_bank)
                self.faiss_index = index

            except Exception as e:
                logger.warning(f"FAISS GPU 설정 실패 ({e}). CPU 모드로 전환합니다.")
                self.faiss_index = faiss.IndexFlatL2(dim)
                self.faiss_index.add(self.memory_bank)
        else:
            logger.info("Faiss 없음: Scikit-Learn KNN 사용.")
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn.fit(self.memory_bank)

    def predict(self, img, score_type="max", kneighbors_batch=None):
        # 배치 차원 추가
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # 특징 추출
        test_feat = self.extract_features(img).numpy()
        test_feat = np.ascontiguousarray(test_feat.astype(np.float32))

        # 검색
        if self.faiss_index is not None:
            distances = []
            if kneighbors_batch:
                for start in range(0, test_feat.shape[0], kneighbors_batch):
                    end = start + kneighbors_batch
                    D, _ = self.faiss_index.search(
                        test_feat[start:end], self.n_neighbors
                    )
                    distances.append(D)
                D = np.concatenate(distances, axis=0)
            else:
                D, _ = self.faiss_index.search(test_feat, self.n_neighbors)
            patch_scores = np.mean(D, axis=1)
        elif self.knn is not None:
            distances = []
            if kneighbors_batch:
                for start in range(0, test_feat.shape[0], kneighbors_batch):
                    end = start + kneighbors_batch
                    D, _ = self.knn.kneighbors(
                        test_feat[start:end], n_neighbors=self.n_neighbors
                    )
                    distances.append(D)
                D = np.concatenate(distances, axis=0)
            else:
                D, _ = self.knn.kneighbors(test_feat)
            patch_scores = np.mean(D, axis=1)
        else:
            raise RuntimeError("모델 미학습 상태")

        # 배치별 점수 계산
        patches_per_img = test_feat.shape[0] // img.shape[0]
        batch_scores = []

        for i in range(img.shape[0]):
            start = i * patches_per_img
            end = (i + 1) * patches_per_img
            scores_in_img = patch_scores[start:end]

            if score_type == "max":
                score = np.max(scores_in_img)
            else:
                score = np.mean(scores_in_img)
            batch_scores.append(float(score))

        return batch_scores


# Backward-compatible alias used by training script
class PatchCoreFromScratch(PatchCoreOptimized):
    pass
