# obsidian_summary.py
import os
import re
import yaml
import glob
import json
import time
import threading
import concurrent.futures
from pathlib import Path
import requests
from functools import partial
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def load_config(config_path: str = 'config.json') -> Dict:
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"설정 파일 '{config_path}'을(를) 로드했습니다.")
        return config
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        print("기본 설정을 사용합니다.")
        return {
            "model": "qwen3:14b",
            "embedding_model": "mxbai-embed-large",
            "vault_paths": [
                "D:\\z-vault\\03. Knowledge",
                "D:\\z-vault\\01. Think & Plan",
                "D:\\z-vault\\02. Projects",
                "D:\\z-vault\\00. News"
            ],
            "skip_categories": ["knowledge", "plan", "main-project"],
            "exclude_directories": [
                "D:\\z-vault\\00. News\\original"
            ],
            "min_content_length": {
                "summary": 50,
                "embedding": 100
            },
            "summary_length": {
                "short": 1,
                "medium": 2,
                "long": 4
            },
            "tag_count": {
                "min": 5,
                "max": 10
            },
            "metadata_fields": [
                "title",
                "category",
                "summary", 
                "tags",
                "source",
                "date"
            ],
            "embedding": {
                "similarity_threshold": 0.7,
                "max_backlinks": 7,
                "jsonl_path": "D:\\z-vault\\.vectors\\embeddings.jsonl"
            },
            "api_settings": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 500
            },
            "parallel_processing": True,
            "num_workers": 8,
            "test_mode": False,
            "force_reprocess": False
        }

def generate_with_ollama(prompt: str, config: Dict) -> str:
    """Ollama API를 사용하여 텍스트를 생성합니다."""
    url = "http://localhost:11434/api/generate"
    
    api_settings = config["api_settings"]
    model = config["model"]
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": api_settings["max_tokens"],
        "stream": False,
        "temperature": api_settings["temperature"],
        "top_p": api_settings["top_p"]
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Ollama API 호출 중 오류 발생: {e}")
        return ""

def test_ollama(config: Dict) -> bool:
    """Ollama 설정이 올바르게 되었는지 간단한 테스트를 수행합니다."""
    try:
        model = config["model"]
        prompt = "Hello, I am a test prompt."
        response = generate_with_ollama(prompt, config)
        print(f"Ollama 테스트 응답: {response}")
        return True
    except Exception as e:
        print(f"Ollama 테스트 중 오류 발생: {e}")
        return False

def split_frontmatter_content(text: str) -> Tuple[Dict, str]:
    """노트에서 프론트매터와 본문 컨텐츠를 분리합니다."""
    if text.startswith('---'):
        # 프론트매터가 있는 경우
        parts = text.split('---', 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            content = parts[2].strip()
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
                if frontmatter is None:  # YAML이 비어있는 경우
                    frontmatter = {}
            except Exception as e:
                print(f"YAML 파싱 오류: {e}")
                frontmatter = {}
            return frontmatter, content
    
    # 프론트매터가 없거나 형식이 잘못된 경우
    return {}, text

def check_process_status(frontmatter: Dict, step: str) -> bool:
    """특정 처리 단계가 완료되었는지 확인합니다."""
    if 'processed' in frontmatter and isinstance(frontmatter['processed'], list):
        for item in frontmatter['processed']:
            if isinstance(item, str) and item.split('#')[0].strip() == step:
                return True
    return False

def update_process_status(frontmatter: Dict, step: str) -> Dict:
    """처리 상태를 업데이트합니다."""
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if 'processed' not in frontmatter:
        frontmatter['processed'] = []
    
    if not isinstance(frontmatter['processed'], list):
        frontmatter['processed'] = []
    
    for i, item in enumerate(frontmatter['processed']):
        if isinstance(item, str) and item.split('#')[0].strip() == step:
            frontmatter['processed'][i] = f"{step} # '{current_date}'"
            return frontmatter
    
    frontmatter['processed'].append(f"{step} # '{current_date}'")
    return frontmatter

def determine_summary_length(content_length: int, config: Dict) -> int:
    """노트 길이에 따라 적절한 요약 길이를 결정합니다."""
    summary_config = config["summary_length"]
    
    if content_length < 500:
        return summary_config["short"]
    elif content_length < 2000:
        return summary_config["medium"]
    else:
        return summary_config["long"]

def clean_tag(tag: str) -> str:
    """태그에서 백틱과 공백을 제거하고 옵시디언에 적합한 형태로 변환합니다."""
    # 백틱 제거
    tag = tag.replace('`', '')
    # 공백을 하이픈으로 변환
    tag = tag.replace(' ', '-')
    # 앞뒤 특수문자 제거
    tag = tag.strip('-_#')
    return tag

def generate_summary_and_tags(content: str, title: str, config: Dict) -> Tuple[str, List[str]]:
    """Ollama를 사용하여 노트 내용에 대한 요약과 태그를 생성합니다."""
    min_content_length = config["min_content_length"]["summary"]
    tag_count = config["tag_count"]
    
    # 컨텐츠가 너무 짧거나 없는 경우
    if len(content.strip()) < min_content_length:
        return "", []
    
    # 노트 길이에 따른 요약 길이 결정
    summary_lines = determine_summary_length(len(content), config)
    
    # 프롬프트 구성 (한국어로 요청)
    prompt = f"""
당신은 옵시디언 노트의 내용을 요약하고 태그를 추출하는 AI 어시스턴트입니다. 
요약문은 핵심 주제를 빠짐없이 포함하되, 문장은 간결하게 마무리하세요. 각 문장은 독립적으로 의미가 통하는 구조로 작성하세요.
다음 내용의 노트에 대해 {summary_lines}줄 길이의 요약과 {tag_count["min"]}-{tag_count["max"]}개의 태그를 생성해주세요.

노트 제목: {title}

노트 내용:
{content[:4000]}

다음 형식으로 출력해주세요:
요약: <요약 내용>
태그: <태그1>, <태그2>, <태그3>, ...

중요: 태그는 공백 없이 하나의 단어 또는 하이픈으로 연결된 단어로 작성해주세요. 백틱(`)을 사용하지 마세요.
"""

    # Ollama 호출
    output = generate_with_ollama(prompt, config)
    
    # 응답 파싱
    summary = ""
    tags = []
    
    summary_match = re.search(r'요약: (.+?)(?=태그:|$)', output, re.DOTALL)
    tags_match = re.search(r'태그: (.+?)$', output, re.DOTALL)
    
    if summary_match:
        summary = summary_match.group(1).strip()
    
    if tags_match:
        tags_text = tags_match.group(1).strip()
        # 쉼표로 구분된 태그 처리
        raw_tags = [tag.strip() for tag in tags_text.split(',')]
        cleaned_tags = []
        
        for tag in raw_tags:
            # 해시태그가 있는 경우 처리
            if '#' in tag:
                hash_tags = [t.strip('# ') for t in tag.split() if t.startswith('#')]
                cleaned_tags.extend([clean_tag(t) for t in hash_tags])
            else:
                cleaned_tags.append(clean_tag(tag))
        
        # 빈 태그 제거
        tags = [tag for tag in cleaned_tags if tag]
    
    return summary, tags

def update_frontmatter(frontmatter: Dict, summary: str, new_tags: List[str]) -> Dict:
    """기존 프론트매터에 요약과 새 태그를 추가합니다."""
    updated_frontmatter = frontmatter.copy()
    
    # 요약 추가 (summary 필드에 저장)
    if summary:
        updated_frontmatter['summary'] = summary
    
    # 태그 업데이트
    if 'tags' in updated_frontmatter and updated_frontmatter['tags']:
        if isinstance(updated_frontmatter['tags'], list):
            # 기존 태그 정리
            existing_tags = []
            for tag in updated_frontmatter['tags']:
                if isinstance(tag, str):
                    existing_tags.append(clean_tag(tag))
            
            # 새 태그와 결합 (중복 제거)
            all_tags = set(existing_tags)
            all_tags.update([clean_tag(tag) for tag in new_tags])
            updated_frontmatter['tags'] = sorted(list(all_tags))
        else:
            # 태그가 문자열 등 다른 형식인 경우 리스트로 변환
            updated_frontmatter['tags'] = [clean_tag(tag) for tag in new_tags]
    else:
        # 태그가 없는 경우 새로 추가
        updated_frontmatter['tags'] = [clean_tag(tag) for tag in new_tags]
    
    # 처리 상태 업데이트
    updated_frontmatter = update_process_status(updated_frontmatter, "summary")
    
    return updated_frontmatter

def should_process_note(frontmatter: Dict, config: Dict) -> bool:
    """노트를 처리해야 하는지 결정합니다."""
    # 설정에서 강제 재처리 옵션 확인
    force_reprocess = config.get("force_reprocess", False)
    if force_reprocess:
        return True
    
    # 이미 요약 처리가 되었는지 확인
    if check_process_status(frontmatter, "summary"):
        return False
    
    return True

def save_note(file_path: str, frontmatter: Dict, content: str) -> None:
    """업데이트된 프론트매터와 컨텐츠로 노트를 저장합니다."""
    # YAML 덤프 옵션 설정 (Flow style 비활성화)
    yaml_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    
    # 최종 노트 내용 조합
    updated_note = f"---\n{yaml_text}---\n{content}"
    
    # 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_note)

def should_process_category(category: str, config: Dict) -> bool:
    """카테고리에 따라 처리 여부를 결정합니다."""
    skip_categories = config["skip_categories"]
    if category in skip_categories:
        return False
    return True

def should_process_file(file_path: str, config: Dict) -> bool:
    """파일이 처리 대상인지 확인합니다."""
    # 제외 디렉토리 확인
    exclude_directories = config.get("exclude_directories", [])
    for exclude_dir in exclude_directories:
        if os.path.normpath(file_path).startswith(os.path.normpath(exclude_dir)):
            return False
    return True

def process_single_file(file_path: str, config: Dict, processed_count: Dict, vault_path: str) -> bool:
    """단일 파일을 처리합니다."""
    file_name = os.path.basename(file_path)
    try:
        # 제외 디렉토리 확인
        if not should_process_file(file_path, config):
            print(f"  - 건너뜀: 제외 디렉토리 내 파일 ({file_name})")
            return False
            
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 프론트매터와 컨텐츠 분리
        frontmatter, note_content = split_frontmatter_content(content)
        
        # 카테고리 확인
        category = frontmatter.get('category', '')
        if not should_process_category(category, config):
            print(f"  - 건너뜀: 처리 제외 카테고리 ({file_name}, {category})")
            return False
        
        # 노트 처리 여부 확인
        if not should_process_note(frontmatter, config):
            print(f"  - 건너뜀: 이미 처리된 노트 ({file_name})")
            return False
        
        # 컨텐츠가 충분히 있는지 확인
        min_content_length = config["min_content_length"]["summary"]
        if len(note_content.strip()) < min_content_length:
            print(f"  - 건너뜀: 내용이 너무 적음 ({file_name}, {len(note_content.strip())}자)")
            return False
        
        # 요약 및 태그 생성
        title = frontmatter.get('title', file_name)
        summary, tags = generate_summary_and_tags(note_content, title, config)
        
        if not summary:
            print(f"  - 건너뜀: 요약 생성 실패 ({file_name})")
            return False
        
        # 프론트매터 업데이트
        updated_frontmatter = update_frontmatter(frontmatter, summary, tags)
        
        # 파일 저장
        save_note(file_path, updated_frontmatter, note_content)
        with processed_count["lock"]:
            print(f"  - 완료: {file_name} 요약 및 태그 추가됨")
        
        # 처리된 파일 수 업데이트 (스레드 안전하게 처리)
        with processed_count["lock"]:
            processed_count["total"] += 1
            
        return True
        
    except Exception as e:
        with processed_count["lock"]:
            print(f"  - 오류: {file_name} - {str(e)}")
            processed_count["error"] += 1
        return False

def process_files_parallel(md_files: List[str], config: Dict, processed_count: Dict, vault_path: str) -> None:
    """파일을 병렬로 처리합니다."""
    # 스레드 안전한 카운터를 위한 락
    if "lock" not in processed_count:
        processed_count["lock"] = threading.Lock()
    
    # 작업자 수 설정
    num_workers = config.get("num_workers", 4)
    print(f"병렬 처리 시작: {num_workers}개 작업자")
    
    # 테스트 모드 확인
    test_mode = config.get("test_mode", False)
    
    if test_mode:
        # 테스트 모드에서는 각 폴더당 하나의 파일만 처리
        folder_processed = set()
        for file_path in md_files:
            folder = os.path.dirname(file_path)
            if folder not in folder_processed:
                if process_single_file(file_path, config, processed_count, vault_path):
                    with processed_count["lock"]:
                        print(f"  - 테스트 모드: {folder}에서 한 파일 처리 완료")
                    folder_processed.add(folder)
                    if len(folder_processed) >= 1:  # 테스트 모드에서는 한 폴더만 처리
                        break
    else:
        # 병렬 처리
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # partial 함수로 나머지 매개변수 고정
            process_func = partial(process_single_file, config=config, processed_count=processed_count, vault_path=vault_path)
            
            # 파일 목록에 대해 병렬 실행
            list(executor.map(process_func, md_files))

def process_vault(vault_path: str, config: Dict, processed_count: Dict) -> None:
    """단일 볼트 경로의 노트를 처리합니다."""
    print(f"처리 중인 볼트: {vault_path}")
    md_files = glob.glob(os.path.join(vault_path, "**/*.md"), recursive=True)
    
    # 처리할 총 파일 수
    total_files = len(md_files)
    print(f"총 {total_files}개 파일 발견")
    
    # 병렬 처리 설정
    parallel = config.get("parallel_processing", False)
    
    if parallel:
        # 병렬 처리
        process_files_parallel(md_files, config, processed_count, vault_path)
    else:
        # 순차 처리
        processed_in_folder = 0
        for i, file_path in enumerate(md_files):
            # 진행 상황 출력
            print(f"[{i+1}/{total_files}] 처리 중: {os.path.basename(file_path)}")
            
            # 제외 디렉토리 확인
            if not should_process_file(file_path, config):
                print(f"  - 건너뜀: 제외 디렉토리 내 파일 ({file_path})")
                continue
                
            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 프론트매터와 컨텐츠 분리
                frontmatter, note_content = split_frontmatter_content(content)
                
                # 카테고리 확인
                category = frontmatter.get('category', '')
                if not should_process_category(category, config):
                    print(f"  - 건너뜀: 처리 제외 카테고리 ({category})")
                    continue
                
                # 노트 처리 여부 확인
                if not should_process_note(frontmatter, config):
                    print(f"  - 건너뜀: 이미 처리된 노트 (processed 포함: {check_process_status(frontmatter, 'summary')})")
                    continue
                
                # 컨텐츠가 충분히 있는지 확인
                min_content_length = config["min_content_length"]["summary"]
                if len(note_content.strip()) < min_content_length:
                    print(f"  - 건너뜀: 내용이 너무 적음 ({len(note_content.strip())}자)")
                    continue
                
                # 요약 및 태그 생성
                title = frontmatter.get('title', os.path.basename(file_path))
                summary, tags = generate_summary_and_tags(note_content, title, config)
                
                if not summary:
                    print(f"  - 건너뜀: 요약 생성 실패")
                    continue
                
                # 프론트매터 업데이트
                updated_frontmatter = update_frontmatter(frontmatter, summary, tags)
                
                # 파일 저장
                save_note(file_path, updated_frontmatter, note_content)
                print(f"  - 완료: 요약 및 태그 추가됨")
                
                # 처리된 파일 수 업데이트
                processed_count["total"] += 1
                
                # 테스트 모드 처리
                test_mode = config.get("test_mode", False)
                if test_mode:
                    processed_in_folder += 1
                    if processed_in_folder >= 1:  # 각 폴더에서 첫 번째 파일만 처리
                        print(f"  - 테스트 모드: {vault_path}에서 한 파일 처리 완료")
                        break
                
                # API 요청 사이에 잠시 대기
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  - 오류: {e}")
                processed_count["error"] += 1

def main():
    """메인 함수: 설정 로드 및 노트 처리를 실행합니다."""
    # 설정 로드
    config = load_config()
    
    # Ollama 테스트
    print(f"Ollama 테스트 중: {config['model']}")
    if not test_ollama(config):
        print("Ollama 설정에 문제가 있습니다. 프로그램을 종료합니다.")
        return
    
    # 처리 통계
    processed_count = {"total": 0, "error": 0, "lock": threading.Lock()}
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 각 볼트 경로 처리
    for vault_path in config["vault_paths"]:
        process_vault(vault_path, config, processed_count)
    
    # 종료 시간 기록 및 소요 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 처리 결과 출력
    print(f"\n처리 완료: 총 {processed_count['total']}개 파일 처리됨 (오류: {processed_count['error']}개)")
    print(f"총 소요 시간: {elapsed_time:.2f}초, 파일당 평균 처리 시간: {elapsed_time/max(processed_count['total'], 1):.2f}초")
    
    # 테스트 모드 안내
    if config.get("test_mode", False):
        print("\n테스트 모드로 실행되었습니다. 전체 파일을 처리하려면 config.json에서 'test_mode'를 false로 설정하세요.")

if __name__ == "__main__":
    main()