# obsidian_processor.py
import os
import subprocess
import time
import sys
import json
from datetime import datetime

def load_config(config_path: str = 'config.json'):
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        return {}

def save_config(config, config_path: str = 'config.json'):
    """설정을 파일에 저장합니다."""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def log_processing(log_message, log_file="obsidian_process.log"):
    """처리 결과를 로그 파일에 기록합니다."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {log_message}\n"
    
    # 로그 파일 추가 모드로 열기
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    # 콘솔에도 출력
    print(log_entry.strip())

def extract_stats(output):
    """스크립트 출력에서 처리 통계를 추출합니다."""
    stats = {
        "total_processed": 0,
        "errors": 0,
        "time_elapsed": 0
    }
    
    # 정규식으로 찾는 방법도 있지만, 간단한 문자열 검색으로 구현
    for line in output.split('\n'):
        if "처리 완료: 총" in line and "개 파일 처리됨" in line:
            try:
                stats["total_processed"] = int(line.split("총")[1].split("개")[0].strip())
                stats["errors"] = int(line.split("오류:")[1].split("개")[0].strip())
            except:
                pass
                
        if "총 소요 시간:" in line:
            try:
                stats["time_elapsed"] = float(line.split("시간:")[1].split("초")[0].strip())
            except:
                pass
                
        if "임베딩 생성 완료:" in line:
            try:
                stats["total_processed"] = int(line.split("완료:")[1].split("개")[0].strip())
            except:
                pass
                
        if "백링크 업데이트 완료:" in line:
            try:
                stats["total_backlinks"] = line.split("완료:")[1].split("개")[0].strip()
            except:
                pass
    
    return stats

def run_script(script_name, description):
    """스크립트를 실행하고 결과를 실시간으로 출력합니다."""
    print(f"\n{'='*50}")
    print(f"   {description} 실행 중...")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    try:
        # 파이썬 스크립트 실행 - 실시간 출력을 위해 capture_output=False
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # Python 출력 인코딩 설정
        
        # 실시간 출력을 위해 capture_output=False 설정
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,  # 직접 콘솔에 출력
            text=True,
            encoding='utf-8',
            env=env,
            check=False  # 오류가 발생해도 예외를 발생시키지 않음
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n{description} 완료 (소요 시간: {elapsed:.2f}초)")
        
        # 로그 작성 (통계 없이 기본 정보만)
        log_message = f"{description} 완료 (소요 시간: {elapsed:.2f}초)"
        log_processing(log_message)
        
        # 종료 코드 확인
        if result.returncode != 0:
            print(f"경고: 스크립트가 비정상 종료되었습니다 (코드: {result.returncode})")
            log_processing(f"경고: 스크립트가 비정상 종료되었습니다 (코드: {result.returncode})")
            return False, {}
        
        return True, {"elapsed_time": elapsed}
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        log_processing(f"오류: {description} 실행 중 예외 발생 - {str(e)}")
        return False, {}

def main():
    """요약 및 임베딩 스크립트를 순차적으로 실행합니다."""
    print("옵시디언 노트 처리 시작...")
    
    # 시작 로그 기록
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_processing(f"===== 옵시디언 노트 처리 시작 ({current_time}) =====")
    
    # 설정 로드
    config = load_config()
    
    # 사용자에게 처리 방식 물어보기
    while True:
        print("""
처리 모드를 선택하세요:
1. 전체 재처리 (모든 노트의 요약과 임베딩을 다시 생성)
2. 증분 요약 처리 (새로운/변경된 노트만 요약 생성 + 전체 임베딩 처리)
3. 요약만 처리 (새로운/변경된 노트만)
4. 전체 임베딩 처리 (임베딩 생성 및 모든 백링크 재계산)
5. 종료
""")
        mode = input("선택 (1-5): ")
        
        if mode in ['1', '2', '3', '4', '5']:
            if mode == '5':
                print("프로그램을 종료합니다.")
                log_processing("프로그램 종료")
                return
            break
        print("잘못된 선택입니다. 다시 시도하세요.")
    
    # 선택한 모드 로그 기록
    mode_descriptions = {
        '1': "전체 재처리 (모든 노트의 요약과 임베딩 재생성)",
        '2': "증분 요약 처리 (새 노트만 요약 + 전체 임베딩 처리)",
        '3': "요약만 처리 (새 노트만)",
        '4': "전체 임베딩 처리 (임베딩 생성 및 백링크 재계산)"
    }
    log_processing(f"선택한 모드: {mode_descriptions[mode]}")
    
    # 설정 백업
    config_backup = config.copy()
    
    try:
        # 모드에 따라 설정 및 실행
        if mode == '1':  # 전체 재처리
            print("모든 노트의 요약과 임베딩을 다시 생성합니다.")
            config['force_reprocess'] = True
            save_config(config)
            
            # 요약 생성
            success1, stats1 = run_script("obsidian_summary.py", "1. 노트 요약 및 태그 생성")
            if not success1:
                print("요약 생성 중 오류가 발생했습니다. 임베딩 처리를 건너뜁니다.")
                log_processing("요약 생성 중 오류로 임베딩 처리 건너뜀")
                return
            
            # 임베딩 생성
            success2, stats2 = run_script("obsidian_embedding.py", "2. 노트 임베딩 및 백링크 생성")
            if not success2:
                print("임베딩 생성 중 오류가 발생했습니다.")
                log_processing("임베딩 생성 중 오류 발생")
                
        elif mode == '2':  # 증분 요약 처리
            print("새로운/변경된 노트만 요약 생성하고, 전체 임베딩을 처리합니다.")
            # 요약은 증분 처리
            config['force_reprocess'] = False
            save_config(config)
            
            # 요약 생성
            success1, stats1 = run_script("obsidian_summary.py", "1. 노트 요약 및 태그 생성")
            if not success1:
                print("요약 생성 중 오류가 발생했습니다. 임베딩 처리를 건너뜁니다.")
                log_processing("요약 생성 중 오류로 임베딩 처리 건너뜀")
                return
            
            # 임베딩은 증분 처리지만 백링크는 전체 재계산
            success2, stats2 = run_script("obsidian_embedding.py", "2. 노트 임베딩 및 백링크 생성")
            if not success2:
                print("임베딩 생성 중 오류가 발생했습니다.")
                log_processing("임베딩 생성 중 오류 발생")
                
        elif mode == '3':  # 요약만 처리
            print("새로운/변경된 노트만 요약 생성합니다.")
            config['force_reprocess'] = False
            save_config(config)
            
            # 요약 생성
            success, stats = run_script("obsidian_summary.py", "노트 요약 및 태그 생성")
            if not success:
                print("요약 생성 중 오류가 발생했습니다.")
                log_processing("요약 생성 중 오류 발생")
                
        elif mode == '4':  # 전체 임베딩 처리
            print("임베딩을 생성하고 모든 백링크를 재계산합니다.")
            config['force_reprocess'] = False  # 임베딩만 증분 처리
            save_config(config)
            
            # 임베딩 생성
            success, stats = run_script("obsidian_embedding.py", "노트 임베딩 및 백링크 생성")
            if not success:
                print("임베딩 생성 중 오류가 발생했습니다.")
                log_processing("임베딩 생성 중 오류 발생")
        
        print("\n선택한 처리가 완료되었습니다!")
        log_processing("===== 처리 완료 =====\n")
    
    finally:
        # 설정 파일 복원
        save_config(config_backup)
    
if __name__ == "__main__":
    main()