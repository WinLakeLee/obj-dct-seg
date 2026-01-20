#!/usr/bin/env python3
"""
현재 폴더의 파일을 순차적으로 1, 2, 3, ...으로 변경
이미지와 라벨 파일 모두 적용
"""

from pathlib import Path
import sys

def rename_files_in_current_folder():
    """현재 폴더의 모든 파일을 순차적으로 1, 2, 3, ...으로 변경"""
    
    current_dir = Path.cwd()
    # .py 파일 제외 (Python 스크립트는 변경하지 않음)
    files = sorted([f for f in current_dir.iterdir() if f.is_file() and f.suffix.lower() != '.py'])
    
    if not files:
        print(f"❌ 파일이 없습니다: {current_dir}")
        return
    
    print(f"📁 {current_dir}")
    print(f"📊 총 {len(files)}개 파일 변경")
    print()
    
    # 1단계: 모든 파일을 임시 이름으로 변경 (안전성을 위해)
    temp_mapping = {}
    for idx, file_path in enumerate(files, 1):
        temp_name = f"__temp_{idx}__"
        temp_path = file_path.parent / temp_name
        try:
            file_path.rename(temp_path)
            temp_mapping[temp_name] = idx
        except Exception as e:
            print(f"❌ {file_path.name}: {e}")
            return
    
    # 2단계: 임시 파일을 최종 이름으로 변경
    for temp_name, idx in sorted(temp_mapping.items()):
        temp_path = current_dir / temp_name
        # 원본 파일 확장자를 유지하기 위해 첫 파일에서 확장자 추출
        if temp_path.exists():
            ext = files[idx-1].suffix.lower()
            new_name = f"{idx}{ext}"
            new_path = current_dir / new_name
            try:
                temp_path.rename(new_path)
                print(f"✅ {files[idx-1].name} → {new_name}")
            except Exception as e:
                print(f"❌ 최종 변경 실패: {e}")
    
    print()
    print("✨ 완료!")

if __name__ == "__main__":
    rename_files_in_current_folder()
