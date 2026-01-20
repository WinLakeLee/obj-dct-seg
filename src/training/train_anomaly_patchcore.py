from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import os

# 1. 데이터 로딩 (NEU 데이터셋 경로 설정)
# Assumes script is run from project root or src/training
# Adjust root to point to data/neu_metal relative to execution
dataset_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "neu_metal")
)

datamodule = Folder(
    name="metal_scratches",
    root=dataset_root,
    normal_dir="train/good",  # 학습용 정상 이미지
    abnormal_dir="test/scratch",  # 테스트용 불량 이미지
    task="segmentation",  # 결함 부위를 색칠해서 보여줌
)

# 2. 모델 생성 (PatchCore)
model = Patchcore(backbone="wide_resnet50_2")

# 3. 학습 및 검증 엔진
engine = Engine(task="segmentation", default_root_dir="outputs/anomalib_patchcore")

# 4. 학습 시작
print("🚀 PatchCore 학습 시작...")
engine.fit(datamodule=datamodule, model=model)

# 5. 테스트
print("🧐 테스트 진행 중...")
test_results = engine.test(datamodule=datamodule, model=model)
