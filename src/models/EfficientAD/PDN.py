import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import logging
import timm

from src.utils.data_utils import build_torch_transform, make_torch_dataloader


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EfficientAD")


# ==========================================
# 1. 모델 아키텍처 (PDN - Patch Description Network)
# ==========================================
class PDN(nn.Module):
    """
    논문에서 제안한 경량화된 특징 추출 네트워크 (Small 버전)
    ImageNet 사전 학습의 효과를 내면서도 훨씬 빠름.
    """

    def __init__(self, out_channels=384):
        super(PDN, self).__init__()
        # EfficientAD는 4x4 Conv와 AvgPool을 적극 사용하여 Aliasing을 방지함
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.avgpool1(x)
        x = self.activation(self.conv2(x))
        x = self.avgpool2(x)
        x = self.activation(self.conv3(x))
        x = self.conv4(x)
        return x


class AutoEncoder(nn.Module):
    """
    논리적 이상(Logical Anomaly)을 탐지하기 위한 보조 네트워크
    """

    def __init__(self, out_channels=384):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 8, stride=1, padding=0),  # Bottleneck
        )
        self.decoder = nn.Sequential(
            # Use 6 upsample steps and 3x3 convs with padding=1 to preserve spatial size
            # Starting from a 1x1 bottleneck, 1 * 2^6 = 64 final spatial resolution
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            # Output 채널을 PDN의 출력 채널과 맞춰서 Student가 학습하기 쉽게 함
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


class TimmTeacher(nn.Module):
    """
    ImageNet으로 사전 학습된 강력한 선생님 (WideResNet-50)
    특징 추출(Feature Extraction)만 수행합니다.
    """

    def __init__(self, model_name="wide_resnet50_2"):
        super(TimmTeacher, self).__init__()
        # features_only=True: 분류기(Classifier)를 떼고 특징만 뽑음
        # out_indices=[1]: 2번째 스테이지의 특징만 사용 (너무 얕지도, 깊지도 않은 적절한 위치)
        self.model = timm.create_model(
            model_name, pretrained=True, features_only=True, out_indices=[1]
        )

        # 파라미터 고정 (학습되지 않도록 Freeze)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()  # 언제나 평가 모드

    def forward(self, x):
        # timm의 features_only 모델은 리스트를 반환함 [feature1, feature2, ...]
        features = self.model(x)
        return features[0]  # 우리가 선택한 스테이지의 특징 맵 반환


# ==========================================
# 2. EfficientAD 전체 모델 클래스
# ==========================================
class EfficientAD:
    def __init__(self, seed=42, out_channels=384, image_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # 시드 고정
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # 1. Teacher (고정된 네트워크)
        # ---------------------------------------------------------
        # [변경 1] Teacher를 Random PDN -> Pretrained WideResNet으로 교체
        # ---------------------------------------------------------
        self.teacher = TimmTeacher(model_name="resnet18").to(self.device)
        self.teacher.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            teacher_out = self.teacher(dummy_input)
            out_channels = teacher_out.shape[1]  # 예: 512
            logger.info(f"🧠 Teacher Model Loaded (Channels: {out_channels})")

        # Teacher 가중치 초기화 (ImageNet Distillation 흉내 - 랜덤이지만 구조적 특성 활용)
        # 실제 논문에서는 ImageNet pre-trained weights를 distillation하지만,
        # 여기서는 랜덤 초기화된 Teacher를 Ground Truth로 삼는 변형(RD4AD 방식)을 차용해 의존성 제거

        # ---------------------------------------------------------
        # [변경 2] Student와 AE의 채널 수를 Teacher에 맞춤
        # ---------------------------------------------------------
        # Student(PDN)는 가볍게 유지하되, 출력층(conv4)만 Teacher와 크기를 맞춥니다.
        self.student = PDN(out_channels=out_channels).to(self.device)
        self.ae = AutoEncoder(out_channels=out_channels).to(self.device)

        # 최적화기 설정 (Teacher는 학습 안 하므로 제외)
        self.optimizer = torch.optim.Adam(
            list(self.student.parameters()) + list(self.ae.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )

        # 정규화 통계 저장소 (크기 맞춤)
        self.teacher_mean = torch.zeros(1, out_channels, 1, 1).to(self.device)
        self.teacher_std = torch.ones(1, out_channels, 1, 1).to(self.device)

    def _normalize_teacher_output(self, teacher_out):
        # [수정] epsilon 추가로 0으로 나누기 방지
        return (teacher_out - self.teacher_mean) / (self.teacher_std + 1e-6)

    def calculated_hard_loss(self, teacher_out, student_out, q=0.999):
        """
        모든 픽셀의 평균을 구하는 대신, 오차가 가장 큰 상위 (1-q)% 픽셀들의 평균만 구함.
        작은 결함을 놓치지 않게 해줌.
        """
        # (Batch, Channel, H, W) -> (Batch, -1)
        diff = (teacher_out - student_out) ** 2
        batch_size = diff.shape[0]
        flatten_diff = diff.view(batch_size, -1)

        # 상위 k개 픽셀 선택 (Hard Negative Mining)
        # q=0.999라면 상위 0.1%의 오차만 학습에 반영
        num_hard_pixels = int(flatten_diff.shape[1] * (1 - q))
        if num_hard_pixels < 1:
            num_hard_pixels = 1

        hard_diff, _ = torch.topk(flatten_diff, k=num_hard_pixels, dim=1)
        return torch.mean(hard_diff)

    def train(self, dataloader, epochs=100):
        logger.info(f"🚀 EfficientAD 학습 시작 (Improved Version)")

        # 1. Teacher 통계 계산 (기존과 동일)
        logger.info("📊 Teacher Output 통계 계산 중...")
        with torch.no_grad():
            outputs = []
            for imgs in dataloader:
                imgs = imgs.to(self.device)
                outputs.append(self.teacher(imgs))
            outputs = torch.cat(outputs, dim=0)
            self.teacher_mean = torch.mean(outputs, dim=[0, 2, 3], keepdim=True)
            self.teacher_std = torch.std(outputs, dim=[0, 2, 3], keepdim=True)
            logger.info("✅ 통계 계산 완료.")

        self.student.train()
        self.ae.train()

        # [개선] 스케줄러 추가 (학습 후반부 미세 조정)
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(epochs * 0.8), gamma=0.1
        )

        for epoch in range(epochs):
            total_loss = 0
            for imgs in dataloader:
                imgs = imgs.to(self.device)

                with torch.no_grad():
                    teacher_out = self.teacher(imgs)
                    teacher_out = self._normalize_teacher_output(teacher_out)

                student_out = self.student(imgs)
                ae_out = self.ae(imgs)

                # Ensure AE output spatial size matches teacher/student (avoid broadcasting warnings)
                if ae_out.shape[2:] != teacher_out.shape[2:]:
                    ae_out = F.interpolate(
                        ae_out, size=teacher_out.shape[2:], mode="bilinear", align_corners=False
                    )

                # --- [핵심 변경] Loss Calculation ---

                # 1. Local Loss: Hard Feature Mining 적용 (q=0.99 ~ 0.999 권장)
                # 전체 평균 대신 오차가 큰 픽셀에 집중하여 '미세 결함' 검출력 상승
                loss_st = self.calculated_hard_loss(teacher_out, student_out, q=0.99)

                # 2. AE Loss: 전체적인 구조 학습은 그대로 MSE 사용 (전체 형상을 봐야 하므로)
                loss_ae = F.mse_loss(ae_out, teacher_out)

                # 3. ST-AE Loss: Student가 AE를 따라하게 함
                loss_st_ae = F.mse_loss(student_out, ae_out.detach())

                # 가중치 조절 (논문에서는 loss_st에 가중치를 1로 두지만,
                # 미세 결함이 중요하다면 loss_st 비중을 높임)
                loss = loss_st + loss_ae + loss_st_ae

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            scheduler.step()  # 스케줄러 업데이트

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.6f}"
                )

        logger.info("🎉 학습 완료!")

    def predict(self, img):
        """
        단일 이미지 추론
        img: Tensor (1, 3, H, W)
        Return: Anomaly Map (H, W), Score (float)
        """
        self.student.eval()
        self.ae.eval()

        img = img.to(self.device)

        with torch.no_grad():
            teacher_out = self.teacher(img)
            teacher_out = self._normalize_teacher_output(teacher_out)

            student_out = self.student(img)
            ae_out = self.ae(img)
            # Upsample AE output to teacher spatial size if needed
            if ae_out.shape[2:] != teacher_out.shape[2:]:
                ae_out = F.interpolate(
                    ae_out, size=teacher_out.shape[2:], mode="bilinear", align_corners=False
                )
            # 1. Local Map: Teacher vs Student 차이
            # 채널 방향으로 평균을 내어 (H, W) 맵 생성
            map_st = torch.mean((teacher_out - student_out) ** 2, dim=1, keepdim=True)

            # 2. Global Map: Teacher vs AE 차이
            map_ae = torch.mean((teacher_out - ae_out) ** 2, dim=1, keepdim=True)

            # 3. 결합
            combined_map = map_st + map_ae

            # 원본 해상도로 Upsample
            anomaly_map = F.interpolate(
                combined_map,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

            # 결과 가공
            anomaly_map = anomaly_map[0, 0].cpu().numpy()
            anomaly_score = np.max(anomaly_map)  # 가장 이상한 부분의 점수

        return anomaly_map, anomaly_score


# ==========================================
# 3. 실행 유틸리티
# ==========================================
def get_dataloader(data_dir, img_size=256, batch_size=16):
    transform = build_torch_transform(resize_size=img_size, crop_size=None, normalize=True)
    return make_torch_dataloader(
        data_dir,
        batch_size=batch_size,
        num_workers=4,
        transform=transform,
        shuffle=True,
        recursive=True,
    )


# ==========================================
# 사용 예시
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 경로 설정
    DATA_PATH = "data/mvtec/bottle/train/good"  # 예시 경로

    if os.path.exists(DATA_PATH):
        # 2. 데이터 로더 준비
        loader = get_dataloader(DATA_PATH, img_size=256, batch_size=8)

        # 3. 모델 초기화 및 학습
        model = EfficientAD(out_channels=384, image_size=256)
        model.train(loader, epochs=50)  # EfficientAD는 빨리 수렴하므로 Epoch 적어도 됨

        # 4. 추론 테스트
        test_img, _ = next(iter(loader))  # 테스트용 이미지 하나 꺼냄
        test_img = test_img[0:1]  # (1, 3, 256, 256)

        a_map, a_score = model.predict(test_img)
        print(f"Detected Anomaly Score: {a_score:.4f}")

        # 시각화 (Matplotlib)
        import matplotlib.pyplot as plt

        plt.imshow(a_map, cmap="jet")
        plt.title(f"Anomaly Map (Score: {a_score:.2f})")
        plt.colorbar()
        plt.show()
    else:
        print(f"경로를 확인해주세요: {DATA_PATH}")
        print("MVTec 데이터셋 경로를 입력하면 바로 작동합니다.")
