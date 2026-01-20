from anomalib.data import Folder
from anomalib.models import Ganomaly
from anomalib.engine import Engine
import os

# 1. 데이터 로딩
dataset_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "neu_metal")
)

datamodule = Folder(
    name="metal_scratches",
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test/scratch",
    task="segmentation",
)

# 2. 모델 생성
model = Ganomaly()

# 3. 학습 및 검증 엔진
engine = Engine(task="segmentation", default_root_dir="outputs/anomalib_gan")

# 4. 학습 시작
print("🚀 GANomaly 학습 시작...")
engine.fit(datamodule=datamodule, model=model)

# 5. 테스트
print("🧐 테스트 진행 중...")
test_results = engine.test(datamodule=datamodule, model=model)
