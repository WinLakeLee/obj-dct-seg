from anomalib.data import Folder
from anomalib.models import Ganomaly
from anomalib.engine import Engine

# 1. 데이터 로딩 (NEU 데이터셋 경로 설정)
datamodule = Folder(
    name="metal_scratches",
    root="./dataset/neu_metal",
    normal_dir="train/good",  # 학습용 정상 이미지
    abnormal_dir="test/scratch",  # 테스트용 불량 이미지
    task="segmentation",  # 결함 부위를 색칠해서 보여줌
)

# 2. 모델 생성 (Ganomaly)
model = Ganomaly()

# 3. 학습 및 검증 엔진
engine = Engine(task="segmentation")

# 4. 학습 시작 (놀랍게도 몇 분이면 끝납니다)
print("🚀 PatchCore 학습 시작...")
engine.fit(datamodule=datamodule, model=model)

# 5. 테스트 (결과 확인)
print("🧐 테스트 진행 중...")
test_results = engine.test(datamodule=datamodule, model=model)
