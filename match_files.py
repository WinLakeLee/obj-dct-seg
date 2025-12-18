#!/usr/bin/env python3
"""
images와 labels 파일을 매칭시켜서 불일치하는 것들 제거
"""

from pathlib import Path

def match_files():
    base = Path("d:/project/404-ai/yolo_training/dataset")
    
    folders = [
        ("classification/train", 104),
        ("classification/valid", None),
        ("classification/test", None),
        ("instance_segmentation/train", 155),
        ("instance_segmentation/valid", 15),
        ("instance_segmentation/test", 6),
    ]
    
    for folder, expected in folders:
        img_dir = base / folder / "images"
        lbl_dir = base / folder / "labels"
        
        if not img_dir.exists():
            continue
        
        images = set(f.stem for f in img_dir.glob('*.*'))
        labels = set(f.stem for f in lbl_dir.glob('*.txt'))
        
        # images와 labels 개수 출력
        print(f"\n📁 {folder}")
        print(f"   images: {len(images)}, labels: {len(labels)}")
        
        # labels가 없는 images 제거
        orphan_images = images - labels
        if orphan_images:
            print(f"   ⚠️  orphan images: {len(orphan_images)}")
            for stem in sorted(orphan_images):
                for f in img_dir.glob(f'{stem}.*'):
                    f.unlink()
                    print(f"      🗑️  {f.name} 제거")
        
        # images가 없는 labels 제거
        orphan_labels = labels - images
        if orphan_labels:
            print(f"   ⚠️  orphan labels: {len(orphan_labels)}")
            for stem in sorted(orphan_labels):
                f = lbl_dir / f'{stem}.txt'
                if f.exists():
                    f.unlink()
                    print(f"      🗑️  {f.name} 제거")
        
        # 최종 개수
        final_images = len(list(img_dir.glob('*.*')))
        final_labels = len(list(lbl_dir.glob('*.txt')))
        print(f"   ✨ 정리 후: images={final_images}, labels={final_labels}")
        
        if expected:
            if final_images == expected and final_labels == expected:
                print(f"   ✅ 기대값({expected})과 일치!")
            else:
                print(f"   ⚠️  기대값({expected})과 불일치")

if __name__ == "__main__":
    match_files()
