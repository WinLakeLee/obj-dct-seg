"""
YOLO + PatchCore 통합 이상 감지 파이프라인

Stage 1: YOLO로 차량 영역 감지
Stage 2: PatchCore로 anomaly detection (스크래치/파손/분리)

사용법:
    python yolo_training/detect_anomaly_pipeline.py --image path/to/image.jpg
    python yolo_training/detect_anomaly_pipeline.py --source path/to/images/ --save-dir results/
"""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from PatchCore.patch_core import PatchCoreOptimized


class ScratchDetectionPipeline:
    def __init__(
        self,
        yolo_model_path='yolo_training/runs/seg_toycar3/weights/last.pt',
        patchcore_checkpoint='models/patchcore_scratch',
        device='cuda',
        conf_threshold=0.25,
        anomaly_threshold=33.08,  # PatchCore 임계값
    ):
        """
        이상 감지 파이프라인 초기화
        
        Args:
            yolo_model_path: YOLO 세그멘테이션 모델 경로
            patchcore_checkpoint: PatchCore 체크포인트 디렉토리
            device: 'cuda' or 'cpu'
            conf_threshold: YOLO 신뢰도 임계값
            anomaly_threshold: PatchCore anomaly 점수 임계값
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # 차량 관련 클래스 ID (data.yaml 기준)
        # 0: objects, 1: car, 2: car_broken_area, 3: car_floor, 4: car_housing, 5: car_scratch, 6: car_separated
        # 비정상 후보: car_broken_area(2), car_separated(6)
        self.car_class_ids = [1, 2, 3, 4, 5, 6]
        self.class_names = {
            1: 'car',
            2: 'car_broken_area',
            3: 'car_floor',
            4: 'car_housing',
            5: 'car_scratch',
            6: 'car_separated',
        }
        
        print(f"🖥️  Device: {self.device}")
        
        # 1. YOLO 모델 로드
        print(f"📦 YOLO 모델 로드: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 2. PatchCore 모델 로드
        print(f"🧠 PatchCore 모델 로드: {patchcore_checkpoint}")
        self.patchcore = self._load_patchcore(patchcore_checkpoint)
        
        # 3. PatchCore용 이미지 전처리
        self.patchcore_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ 파이프라인 초기화 완료!")
    
    def _load_patchcore(self, checkpoint_dir):
        """PatchCore 모델 로드"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # 메모리 뱅크 로드
        memory_bank_path = checkpoint_dir / 'memory_bank.npy'
        if not memory_bank_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")
        
        # 메타데이터 로드
        import json
        meta_path = checkpoint_dir / 'meta.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 모델 생성
        model = PatchCoreOptimized(
            backbone_name='wide_resnet50_2',
            sampling_ratio=meta['sampling_ratio'],
            use_fp16=meta['use_fp16'],
        ).to(self.device)
        
        # 메모리 뱅크 로드
        model.memory_bank = np.load(str(memory_bank_path))
        model.n_neighbors = meta['n_neighbors']
        model._build_index()
        
        return model
    
    def detect_car_regions(self, image_path):
        """
        YOLO로 차량 영역 감지
        
        Returns:
            list of dict: [{'bbox': [x1,y1,x2,y2], 'conf': score, 'class_id': id}, ...]
        """
        results = self.yolo_model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            verbose=False
        )
        
        car_regions = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    
                    # 차량 관련 클래스만 처리
                    if class_id in self.car_class_ids:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        
                        car_regions.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'class_id': class_id
                        })
        
        return car_regions
    
    def detect_anomaly_in_region(self, image, bbox):
        """크롭된 차량 영역에서 PatchCore로 anomaly 점수 계산"""
        x1, y1, x2, y2 = bbox
        
        # 크롭
        cropped = image[y1:y2, x1:x2]
        
        # BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        
        # 전처리 및 배치 차원 추가
        img_tensor = self.patchcore_transform(pil_img).unsqueeze(0)
        
        # PatchCore 추론
        scores = self.patchcore.predict(img_tensor, score_type='max')
        anomaly_score = scores[0]
        
        is_anomaly = anomaly_score >= self.anomaly_threshold

        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'threshold': self.anomaly_threshold,
        }
    
    def process_image(self, image_path, save_path=None):
        """
        단일 이미지 처리
        
        Args:
            image_path: 입력 이미지 경로
            save_path: 결과 저장 경로 (None이면 표시만)
        
        Returns:
            dict: 감지 결과
        """
        print(f"\n🔍 이미지 처리 중: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        result_image = image.copy()
        
        # Stage 1: YOLO로 차량 영역 감지
        car_regions = self.detect_car_regions(image_path)
        print(f"   📦 감지된 차량 영역: {len(car_regions)}개")
        
        results = {
            'image_path': str(image_path),
            'car_regions': [],
            'anomaly_detected': False,
            'scratch_detected': False,
            'broken_detected': False,
            'separated_detected': False,
        }
        
        # Stage 2: 각 차량 영역에서 스크래치 검사
        for i, region in enumerate(car_regions):
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            
            # PatchCore로 anomaly 감지 (스크래치/파손/분리 공통 임계값)
            anomaly_result = self.detect_anomaly_in_region(image, bbox)
            
            cls_id = region['class_id']
            cls_name = self.class_names.get(cls_id, f'class_{cls_id}')

            # 결함 판정 로직
            is_broken_yolo = cls_id == 2
            is_separated_yolo = cls_id == 6
            is_anomaly_pc = anomaly_result['is_anomaly']

            region_result = {
                'bbox': bbox,
                'yolo_conf': region['conf'],
                'class_id': cls_id,
                'class_name': cls_name,
                'anomaly': anomaly_result,
                'broken_by_yolo': is_broken_yolo,
                'separated_by_yolo': is_separated_yolo,
                'anomaly_by_patchcore': is_anomaly_pc,
            }
            results['car_regions'].append(region_result)
            
            # 시각화
            is_defect = is_broken_yolo or is_separated_yolo or is_anomaly_pc
            color = (0, 0, 255) if is_defect else (0, 255, 0)  # 빨강: 결함, 초록: 정상
            thickness = 3 if is_defect else 2
            
            # Bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨
            if is_broken_yolo:
                label = f"broken(yolo)|{anomaly_result['score']:.1f}"
            elif is_separated_yolo:
                label = f"separated(yolo)|{anomaly_result['score']:.1f}"
            else:
                label = f"{cls_name}|{anomaly_result['score']:.1f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if is_anomaly_pc:
                results['anomaly_detected'] = True
            if is_broken_yolo or is_anomaly_pc:
                results['broken_detected'] = True
            if is_separated_yolo:
                results['separated_detected'] = True

            if is_defect:
                print(f"   ⚠️  영역 {i+1}: 결함 감지! (cls={cls_name}, 점수={anomaly_result['score']:.2f})")
            else:
                print(f"   ✅ 영역 {i+1}: 정상 (cls={cls_name}, 점수={anomaly_result['score']:.2f})")
        
        # 결과 저장 또는 표시
        if save_path:
            cv2.imwrite(str(save_path), result_image)
            print(f"   💾 결과 저장: {save_path}")
        
        return results, result_image
    
    def process_directory(self, source_dir, save_dir=None):
        """
        디렉토리 내 모든 이미지 처리
        
        Args:
            source_dir: 입력 이미지 디렉토리
            save_dir: 결과 저장 디렉토리
        """
        source_dir = Path(source_dir)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 찾기
        image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        
        print(f"\n📁 디렉토리 처리: {source_dir}")
        print(f"   이미지 수: {len(image_files)}")
        
        all_results = []
        scratch_count = 0
        
        for img_path in image_files:
            save_path = save_dir / img_path.name if save_dir else None
            result, _ = self.process_image(img_path, save_path)
            all_results.append(result)
            
            if result['scratch_detected']:
                scratch_count += 1
        
        # 요약
        print(f"\n📊 처리 완료:")
        print(f"   전체 이미지: {len(image_files)}")
        print(f"   스크래치 감지: {scratch_count}")
        print(f"   정상: {len(image_files) - scratch_count}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='YOLO + PatchCore 이상 감지 파이프라인')
    parser.add_argument('--image', type=str, help='단일 이미지 경로')
    parser.add_argument('--source', type=str, help='이미지 디렉토리 경로')
    parser.add_argument('--yolo-model', type=str, 
                        default='yolo_training/runs/seg_toycar3/weights/last.pt',
                        help='YOLO 모델 경로')
    parser.add_argument('--patchcore-checkpoint', type=str,
                        default='models/patchcore_scratch',
                        help='PatchCore 체크포인트 디렉토리')
    parser.add_argument('--save-dir', type=str, help='결과 저장 디렉토리')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO 신뢰도 임계값')
    parser.add_argument('--anomaly-threshold', type=float, default=33.08, help='PatchCore anomaly 임계값')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = ScratchDetectionPipeline(
        yolo_model_path=args.yolo_model,
        patchcore_checkpoint=args.patchcore_checkpoint,
        device=args.device,
        conf_threshold=args.conf,
        anomaly_threshold=args.anomaly_threshold,
    )
    
    # 단일 이미지 처리
    if args.image:
        results, result_img = pipeline.process_image(args.image, args.save_dir)
        
        # 결과 출력
        if results['scratch_detected']:
            print(f"\n⚠️  최종 결과: 스크래치 감지됨!")
        else:
            print(f"\n✅ 최종 결과: 정상")
    
    # 디렉토리 처리
    elif args.source:
        pipeline.process_directory(args.source, args.save_dir)
    
    else:
        print("❌ --image 또는 --source를 지정하세요.")
        parser.print_help()


if __name__ == '__main__':
    main()