import os
import re
import yaml
import glob
import json
import time
import threading
import concurrent.futures
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests
from scipy.spatial.distance import cosine
from functools import partial

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
            "model": "gemma3:12b",
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

def get_embedding(text: str, model: str) -> List[float]:
    """Ollama API를 사용하여 텍스트의 임베딩을 생성합니다."""
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("embedding", [])
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return []

def test_embedding(model: str) -> bool:
    """임베딩 API가 올바르게 작동하는지 테스트합니다."""
    try:
        test_text = "This is a test for embedding"
        embedding = get_embedding(test_text, model)
        if embedding and len(embedding) > 0:
            print(f"임베딩 테스트 성공: 모델={model}, 차원={len(embedding)}")
            return True
        else:
            print("임베딩 반환값이 비어 있습니다.")
            return False
    except Exception as e:
        print(f"임베딩 테스트 중 오류 발생: {e}")
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

def prepare_embedding_text(frontmatter: Dict, title: str, config: Dict) -> str:
    """임베딩에 사용할 텍스트를 준비합니다."""
    parts = []
    metadata_fields = config.get("metadata_fields", ["title", "category", "summary", "tags"])
    
    # 제목은 필수 포함
    if title:
        parts.append(f"제목: {title}")
    
    # 설정에 지정된 메타데이터 필드 추가
    for field in metadata_fields:
        if field == "title":  # 이미 처리됨
            continue
            
        value = frontmatter.get(field)
        if value:
            if field == "tags" and isinstance(value, list):
                value = ', '.join(value)
            parts.append(f"{field}: {value}")
    
    return '\n'.join(parts)

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """두 임베딩 벡터 간의 코사인 유사도를 계산합니다."""
    if not embedding1 or not embedding2:
        return 0.0
    
    # 코사인 유사도 계산 (1 - 코사인 거리)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def choose_quote_style(text: str) -> str:
    """텍스트 내용에 따라 최적의 따옴표 스타일을 선택합니다."""
    if "'" in text:
        # 작은따옴표가 포함된 경우 큰따옴표로 감싸기
        # 내부에 큰따옴표가 있다면 이스케이프 처리
        escaped_text = text.replace('"', '\\"')
        return f'"{escaped_text}"'
    else:
        # 작은따옴표가 없는 경우 작은따옴표로 감싸기
        return f"'{text}'"

def sanitize_filename(filename: str) -> str:
    """
    Obsidian 파일명에서 링크 작동에 방해되는 특수문자만 제거합니다.
    작은따옴표는 제거하지 않고 유지합니다.
    """
    if not isinstance(filename, str):
        return str(filename)
        
    # 제거할 특수문자 목록 (최소화)
    invalid_chars = r'#\^[]|'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    
    # 연속된 공백을 하나로 치환
    filename = re.sub(r'\s+', ' ', filename)
    
    # 앞뒤 공백 제거
    filename = filename.strip()
    
    return filename

def escape_yaml_string(text: str) -> str:
    """YAML 문자열 내부의 작은따옴표를 이스케이프합니다."""
    if not isinstance(text, str):
        return str(text)
    return text.replace("'", "''")

def extract_wikilink_content(text: str) -> Tuple[str, str]:
    """
    위키링크 패턴에서 내부 텍스트와 표시 텍스트를 추출합니다.
    [[링크|표시텍스트]] 형식 지원
    """
    # 기본 위키링크 패턴: [[내용]]
    pattern = r'\[\[(.*?)\]\]'
    match = re.search(pattern, text)
    
    if not match:
        return text, text
    
    link_content = match.group(1)
    
    # 파이프로 구분된 위키링크: [[링크|표시텍스트]]
    if '|' in link_content:
        parts = link_content.split('|', 1)
        link = parts[0].strip()
        display = parts[1].strip()
        return link, display
    
    return link_content, link_content

def has_wikilink_pattern(text: str) -> bool:
    """문자열에 위키링크 패턴([[...]])이 있는지 정확히 확인합니다."""
    if not isinstance(text, str):
        return False
    pattern = r'\[\[.*?\]\]'
    return bool(re.search(pattern, text))

def process_frontmatter_value(key: str, value, sanitize_func=None, extract_func=None):
    """
    프론트매터 값을 처리하고 위키링크가 포함된 경우 적절히 포맷팅합니다.
    """
    if sanitize_func is None:
        sanitize_func = lambda x: x
    if extract_func is None:
        extract_func = lambda x: x
    
    if isinstance(value, list):
        # 리스트인 경우
        formatted_list = []
        for item in value:
            if isinstance(item, str):
                # 위키링크 패턴이 있는지 확인
                if has_wikilink_pattern(item):
                    # 위키링크 내용 추출 및 정리
                    link_text, _ = extract_wikilink_content(item)
                    clean_text = sanitize_func(link_text)
                    escaped_text = escape_yaml_string(clean_text)
                    
                    # 이미 완전한 위키링크 형식인 경우
                    if item.strip().startswith('[[') and item.strip().endswith(']]'):
                        formatted_list.append(f"'[[{escaped_text}]]'")
                    else:
                        # 텍스트 내에 위키링크가 있는 경우, 원본 유지하며 이스케이프
                        escaped_item = escape_yaml_string(item)
                        formatted_list.append(f"'{escaped_item}'")
                elif key == 'processed' and '#' in item:
                    # processed 필드는 특별 처리
                    formatted_list.append(item)
                else:
                    # 일반 문자열
                    escaped_item = escape_yaml_string(item)
                    formatted_list.append(f"'{escaped_item}'")
            else:
                formatted_list.append(item)
        return formatted_list
    elif isinstance(value, str):
        # 문자열인 경우
        if has_wikilink_pattern(value):
            # 위키링크 내용 추출 및 정리
            link_text, _ = extract_wikilink_content(value)
            clean_text = sanitize_func(link_text)
            escaped_text = escape_yaml_string(clean_text)
            
            # 이미 완전한 위키링크 형식인 경우
            if value.strip().startswith('[[') and value.strip().endswith(']]'):
                return f"'[[{escaped_text}]]'"
            else:
                # 텍스트 내에 위키링크가 있는 경우, 원본 유지하면서 따옴표 추가
                escaped_value = escape_yaml_string(value)
                return f"'{escaped_value}'"
        else:
            # 공백이나 특수문자가 포함된 경우 따옴표로 감싸기
            escaped_value = escape_yaml_string(value)
            if any(c in value for c in " ,:-"):
                return f"'{escaped_value}'"
            return value
    else:
        # 다른 타입이면 그대로 반환
        return value

def format_frontmatter(frontmatter: Dict, sanitize_func=None, extract_func=None) -> str:
    """
    프론트매터 전체를 포맷팅합니다.
    """
    if sanitize_func is None:
        sanitize_func = sanitize_filename
    if extract_func is None:
        extract_func = lambda x: extract_wikilink_content(x)[0]
    
    # 순서를 유지하도록 필드 정렬
    ordered_keys = []
    
    # title이 있으면 맨 앞에 배치
    if 'title' in frontmatter:
        ordered_keys.append('title')
    
    # category, summary, tags 등 중요 필드 배치
    important_fields = ['category', 'summary', 'tags', 'source', 'date']
    for field in important_fields:
        if field in frontmatter and field not in ordered_keys:
            ordered_keys.append(field)
    
    # 그 외 필드들 (processed와 related_notes 제외)
    for key in frontmatter.keys():
        if key not in ordered_keys + ['processed', 'related_notes']:
            ordered_keys.append(key)
    
    # processed 필드는 마지막에 배치
    if 'processed' in frontmatter:
        ordered_keys.append('processed')
    
    # 필드별로 포맷팅
    formatted_lines = []
    for key in ordered_keys:
        value = frontmatter[key]
        
        # 값 처리 (위키링크 감지 및 포맷팅)
        processed_value = process_frontmatter_value(key, value, sanitize_func, extract_func)
        
        # YAML 형식으로 포맷팅
        if isinstance(processed_value, list):
            formatted_lines.append(f"{key}:")
            for item in processed_value:
                if isinstance(item, str) and not item.startswith('-'):
                    formatted_lines.append(f"  - {item}")
                else:
                    formatted_lines.append(f"  {item}")
        else:
            formatted_lines.append(f"{key}: {processed_value}")
    
    return '\n'.join(formatted_lines)

def process_vault_embeddings(config: Dict) -> Dict[str, Dict]:
    """볼트 내 모든 노트의 임베딩을 생성하고 저장합니다."""
    print("노트 임베딩 처리 시작...")
    embedding_model = config.get("embedding_model", "mxbai-embed-large")
    
    # 테스트 모드 확인
    test_mode = config.get("test_mode", False)
    if test_mode:
        print("테스트 모드로 실행됩니다. 각 폴더당 하나의 파일만 처리합니다.")
    
    # 모든 노트 파일 경로 수집
    all_notes = []
    for vault_path in config["vault_paths"]:
        print(f"볼트 스캔 중: {vault_path}")
        md_files = glob.glob(os.path.join(vault_path, "**/*.md"), recursive=True)
        
        # 제외 디렉토리 필터링
        filtered_files = []
        for file_path in md_files:
            exclude = False
            for exclude_dir in config.get("exclude_directories", []):
                if os.path.normpath(file_path).startswith(os.path.normpath(exclude_dir)):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(file_path)
        
        all_notes.extend(filtered_files)
    
    print(f"총 {len(all_notes)}개 노트 발견")
    
    # 테스트 모드인 경우 각 폴더당 하나의 파일만 처리
    if test_mode:
        processed_folders = set()
        filtered_notes = []
        for file_path in all_notes:
            folder = os.path.dirname(file_path)
            if folder not in processed_folders:
                filtered_notes.append(file_path)
                processed_folders.add(folder)
        all_notes = filtered_notes
        print(f"테스트 모드: {len(all_notes)}개 폴더에서 각 1개 파일 처리")
    
    # 노트 정보 및 임베딩 저장용 딕셔너리
    notes_data = {}
    min_content_length = config.get("min_content_length", {}).get("embedding", 100)
    force_reprocess = config.get("force_reprocess", False)
    
    # ==== 중요: 기존 임베딩 데이터 로드 ====
    existing_data = {}
    jsonl_path = config.get("embedding", {}).get("jsonl_path", "embeddings.jsonl")
    
    if os.path.exists(jsonl_path):
        try:
            print(f"기존 임베딩 데이터를 로드합니다: {jsonl_path}")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        path = data.get('path')
                        if path and os.path.exists(path):  # 존재하는 파일만 포함
                            existing_data[path] = data
                    except json.JSONDecodeError:
                        continue
            print(f"기존 임베딩 데이터 {len(existing_data)}개 로드됨")
        except Exception as e:
            print(f"기존 임베딩 데이터 로드 실패: {e}")
    
    # 병렬 처리 설정
    parallel = config.get("parallel_processing", False)
    num_workers = config.get("num_workers", 4)
    
    
    def process_single_note(file_path):
        """단일 노트 처리 및 임베딩 생성"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 프론트매터와 본문 분리
            frontmatter, note_content = split_frontmatter_content(content)
            
            # 카테고리 확인
            category = frontmatter.get('category', '')
            if category in config.get("skip_categories", []):
                print(f"  - 건너뜀: 처리 제외 카테고리 ({os.path.basename(file_path)}, {category})")
                return None
            
            # 이미 임베딩 되었는지 확인 (force_reprocess가 true면 무시)
            if not force_reprocess and check_process_status(frontmatter, "embedding"):
                # 기존 임베딩 데이터가 있으면 재사용
                if file_path in existing_data:
                    print(f"  - 기존 임베딩 재사용 ({os.path.basename(file_path)})")
                    return (file_path, existing_data[file_path])
                print(f"  - 건너뜀: 이미 임베딩 처리됨 ({os.path.basename(file_path)})")
                return None
            
            # 요약이 있는지 확인
            if 'summary' not in frontmatter:
                print(f"  - 건너뜀: 요약 없음 ({os.path.basename(file_path)})")
                return None
            
            # 노트 길이 확인
            if len(note_content.strip()) < min_content_length:
                print(f"  - 건너뜀: 내용이 너무 적음 ({os.path.basename(file_path)}, {len(note_content.strip())}자)")
                return None
            
            # 임베딩 생성을 위한 텍스트 준비
            title = frontmatter.get('title', os.path.basename(file_path))
            embedding_text = prepare_embedding_text(frontmatter, title, config)
            
            # 임베딩 생성
            embedding_vector = get_embedding(embedding_text, embedding_model)
            
            if not embedding_vector:
                print(f"  - 건너뜀: 임베딩 생성 실패 ({os.path.basename(file_path)})")
                return None
            
            # 프론트매터 업데이트
            updated_frontmatter = update_process_status(frontmatter, "embedding")
            
            # 파일 저장
            formatted_frontmatter = format_frontmatter(updated_frontmatter, sanitize_filename, extract_wikilink_content)
            updated_note = f"---\n{formatted_frontmatter}\n---\n{note_content}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_note)
            
            # 파일 시간 속성 추출
            try:
                # 생성 시간
                created_timestamp = os.path.getctime(file_path)
                created_date = datetime.fromtimestamp(created_timestamp).isoformat()
                
                # 수정 시간
                modified_timestamp = os.path.getmtime(file_path)
                modified_date = datetime.fromtimestamp(modified_timestamp).isoformat()
                
                # 접근 시간
                accessed_timestamp = os.path.getatime(file_path)
                accessed_date = datetime.fromtimestamp(accessed_timestamp).isoformat()
                
                # 파일 크기 (바이트)
                file_size = os.path.getsize(file_path)
            except Exception as e:
                print(f"  - 경고: 파일 속성 추출 중 오류 발생 ({os.path.basename(file_path)}): {e}")
                created_date = modified_date = accessed_date = None
                file_size = 0
            
            # 노트 정보 저장
            note_info = {
                'path': file_path,
                'title': title,
                'category': frontmatter.get('category', ''),
                'embedding': embedding_vector,
                'created_date': created_date,
                'modified_date': modified_date,
                'accessed_date': accessed_date,
                'file_size': file_size
            }
            
            # 설정에 지정된 메타데이터 추가
            for field in config.get("metadata_fields", []):
                if field in frontmatter and field not in note_info:
                    note_info[field] = frontmatter[field]
            
            print(f"  - 완료: 임베딩 생성 ({os.path.basename(file_path)})")
            return (file_path, note_info)
            
        except Exception as e:
            print(f"  - 오류: {os.path.basename(file_path)} - {str(e)}")
            return None
            return None
    
    # 노트 처리 (병렬 또는 순차)
    if parallel:
        print(f"병렬 처리 시작: {num_workers}개 작업자")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_note, all_notes))
            for result in results:
                if result:
                    file_path, note_info = result
                    notes_data[file_path] = note_info
    else:
        for file_path in all_notes:
            result = process_single_note(file_path)
            if result:
                file_path, note_info = result
                notes_data[file_path] = note_info
    
    print(f"새 임베딩 생성 완료: {len(notes_data)}개 노트")
    
    # ==== 중요: 기존 데이터와 새 데이터 병합 ====
    # 기존 임베딩 중 아직 notes_data에 없는 것들 추가
    for path, data in existing_data.items():
        if path not in notes_data and os.path.exists(path):
            if 'embedding' in data:  # 임베딩 데이터가 있는지 확인
                notes_data[path] = data
    
    print(f"통합된 임베딩 데이터: {len(notes_data)}개 노트 (새 임베딩: {len(notes_data) - len(existing_data)}, 기존 임베딩: {len(existing_data)})")
    
    # 임베딩 데이터 저장
    jsonl_path = config.get("embedding", {}).get("jsonl_path", "embeddings.jsonl")
    
    # 디렉토리 확인 및 생성
    jsonl_dir = os.path.dirname(jsonl_path)
    if jsonl_dir and not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for path, note_info in notes_data.items():
            # 임베딩 벡터를 복사하고 datetime 객체 처리
            note_data = {}
            for k, v in note_info.items():
                if k == 'embedding':
                    continue
                elif isinstance(v, datetime):
                    note_data[k] = v.strftime('%Y-%m-%d')
                else:
                    note_data[k] = v
            note_data['embedding'] = note_info['embedding']
            f.write(json.dumps(note_data, ensure_ascii=False) + '\n')
    
    print(f"임베딩 데이터 저장 완료: {jsonl_path}")
    
    return notes_data

def find_related_notes(notes_data: Dict[str, Dict], config: Dict) -> Dict[str, List[Dict]]:
    """각 노트에 대한 관련 노트를 찾아 유사도를 계산합니다."""
    print("관련 노트 찾기 시작...")
    
    # 임베딩 관련 설정
    embedding_config = config.get("embedding", {})
    similarity_threshold = embedding_config.get("similarity_threshold", 0.7)
    max_backlinks = embedding_config.get("max_backlinks", 7)
    
    print(f"현재 설정: similarity_threshold={similarity_threshold}, max_backlinks={max_backlinks}")
    
    # 각 노트별 관련 노트 저장용 딕셔너리
    related_notes = {}
    
    # 모든 노트 쌍에 대해 유사도 계산
    note_paths = list(notes_data.keys())
    total_comparisons = len(note_paths) * (len(note_paths) - 1) // 2
    
    print(f"총 {total_comparisons}개 비교 수행 중... (노트 수: {len(note_paths)})")
    
    comparison_count = 0
    for i, source_path in enumerate(note_paths):
        source_data = notes_data[source_path]
        source_embedding = source_data.get('embedding', [])
        
        if not source_embedding:
            continue
        
        # 현재 노트의 관련 노트 목록
        source_related = []
        
        for j, target_path in enumerate(note_paths):
            # 자기 자신과의 비교 제외
            if i == j:
                continue
                
            target_data = notes_data[target_path]
            target_embedding = target_data.get('embedding', [])
            
            if not target_embedding:
                continue
            
            # 유사도 계산
            similarity = calculate_similarity(source_embedding, target_embedding)
            comparison_count += 1
            
            # 임계값 이상인 경우 관련 노트로 추가
            if similarity >= similarity_threshold:
                related_info = {
                    'path': target_path,
                    'title': target_data.get('title', os.path.basename(target_path)),
                    'similarity': similarity
                }
                
                # 카테고리 정보 추가
                if 'category' in target_data:
                    related_info['category'] = target_data['category']
                
                # 추가 메타데이터 (뉴스인 경우 source와 date 등)
                for field in ['source', 'date']:
                    if field in target_data:
                        related_info[field] = target_data[field]
                
                source_related.append(related_info)
        
        # 유사도 기준으로 정렬하고 최대 갯수만큼 유지
        if source_related:
            source_related.sort(key=lambda x: x['similarity'], reverse=True)
            original_count = len(source_related)
            source_related = source_related[:max_backlinks]
            print(f"  - {os.path.basename(source_path)}: 관련 노트 {original_count}개 중 {len(source_related)}개 선택 (max_backlinks={max_backlinks})")
            related_notes[source_path] = source_related
        
        # 진행 상황 출력
        if (i + 1) % 10 == 0 or i == len(note_paths) - 1:
            print(f"  진행: {i+1}/{len(note_paths)} 노트 처리 중... ({comparison_count}/{total_comparisons} 비교 완료)")
    
    print(f"관련 노트 탐색 완료: {len(related_notes)}개 노트에 백링크 생성 가능")
    return related_notes

def update_notes_with_backlinks(related_notes: Dict[str, List[Dict]], config: Dict) -> None:
    """백링크 정보로 노트들을 업데이트합니다."""
    print("백링크 정보로 노트 업데이트 시작...")
    
    # 임베딩 관련 설정 출력
    embedding_config = config.get("embedding", {})
    max_backlinks = embedding_config.get("max_backlinks", 7)
    print(f"설정된 max_backlinks: {max_backlinks}")
    
    # 병렬 처리 설정
    parallel = config.get("parallel_processing", False)
    num_workers = config.get("num_workers", 4)
    force_reprocess = config.get("force_reprocess", False)
    
    if force_reprocess:
        print("강제 재처리 모드: 이미 백링크가 있는 노트도 다시 처리합니다.")
    

    def update_single_note(file_path):
        """단일 노트에 백링크 정보를 추가합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 프론트매터와 본문 분리
            if not content.startswith('---'):
                print(f"  - 건너뜀: 프론트매터 없음 ({os.path.basename(file_path)})")
                return (file_path, False)
                
            parts = content.split('---', 2)
            if len(parts) < 3:
                print(f"  - 건너뜀: 잘못된 프론트매터 형식 ({os.path.basename(file_path)})")
                return (file_path, False)
                
            frontmatter_text = parts[1].strip()
            note_content = parts[2].strip()
            
            # YAML 파싱
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
                if frontmatter is None:
                    frontmatter = {}
            except Exception as e:
                print(f"  - 오류: YAML 파싱 실패 ({os.path.basename(file_path)}) - {str(e)}")
                return (file_path, False)
            
            # 이미 백링크가 추가되었는지 확인 (force_reprocess가 true면 무시)
            force_reprocess = config.get("force_reprocess", False)
            if not force_reprocess and check_process_status(frontmatter, "backlinks"):
                print(f"  - 건너뜀: 이미 백링크 처리됨 ({os.path.basename(file_path)})")
                return (file_path, False)
            
            # 관련 노트 정보
            related = related_notes.get(file_path, [])
            
            if not related:
                print(f"  - 건너뜀: 관련 노트 없음 ({os.path.basename(file_path)})")
                return (file_path, False)
            
            # 관련 노트 형식화 - 파일 이름 기준으로 변경
            related_items = []
            for item in related:
                # 파일 경로에서 파일 이름 추출
                file_name = os.path.basename(item.get('path', ''))
                # 확장자 제거
                if file_name.endswith('.md'):
                    file_name = file_name[:-3]
                
                # 특수문자 제거 (링크 작동에 방해되는 최소한의 문자만)
                file_name = sanitize_filename(file_name)
                
                similarity = item.get('similarity', 0)
                
                # 위키링크 생성 및 적절한 따옴표 스타일 선택
                link_text = f"[[{file_name}]]"
                formatted_link = choose_quote_style(link_text)
                
                # 백링크 형식으로 포맷팅
                related_items.append(f"- {formatted_link} # similarity: {similarity:.2f}")
            
            # 프론트매터에서 related_notes 제거 (있으면)
            if 'related_notes' in frontmatter:
                del frontmatter['related_notes']
            
            # 처리 상태 업데이트
            frontmatter = update_process_status(frontmatter, "backlinks")
            
            # 프론트매터 포맷팅 (모든 필드에 대해 일관된 처리)
            formatted_frontmatter = format_frontmatter(frontmatter, sanitize_filename, lambda x: extract_wikilink_content(x)[0])
            
            # 관련 노트 추가
            formatted_frontmatter += "\nrelated_notes:"
            for item in related_items:
                formatted_frontmatter += f"\n  {item}"
                
            # 노트 컨텐츠 업데이트
            updated_note = f"---\n{formatted_frontmatter}\n---\n{note_content}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_note)
                
            print(f"  - 완료: 백링크 {len(related_items)}개 추가 ({os.path.basename(file_path)})")
            return (file_path, True)
                
        except Exception as e:
            print(f"  - 오류: {os.path.basename(file_path)} - {str(e)}")
            return (file_path, False)
    
    # 업데이트할 노트 목록
    items_to_update = list(related_notes.keys())
    
    if parallel:
        print(f"병렬 처리 시작: {num_workers}개 작업자")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(update_single_note, items_to_update))
            success_count = sum(1 for _, success in results if success)
            print(f"백링크 업데이트 완료: {success_count}/{len(items_to_update)}개 노트 성공")
    else:
        success_count = 0
        for i, file_path in enumerate(items_to_update):
            _, success = update_single_note(file_path)
            if success:
                success_count += 1
            
            # 진행 상황 출력
            if (i + 1) % 10 == 0 or i == len(items_to_update) - 1:
                print(f"  진행: {i+1}/{len(items_to_update)} 노트 업데이트 중... ({success_count} 성공)")
    
    print(f"백링크 업데이트 완료: {success_count}/{len(items_to_update)}개 노트 성공")

def main():
    """메인 함수: 임베딩 생성 및 백링크 추가를 실행합니다."""
    # 설정 로드
    config = load_config()
   
    # 임베딩 모델 테스트
    embedding_model = config.get("embedding_model", "mxbai-embed-large")
    print(f"임베딩 모델 테스트 중: {embedding_model}")
    if not test_embedding(embedding_model):
        print("임베딩 모델 설정에 문제가 있습니다. 프로그램을 종료합니다.")
        return
   
    # 시작 시간 기록
    start_time = time.time()
   
    # 테스트 모드 확인
    test_mode = config.get("test_mode", False)
    if test_mode:
        print("테스트 모드로 실행됩니다.")
    
    # 임베딩 설정 확인 및 출력
    embedding_config = config.get("embedding", {})
    max_backlinks = embedding_config.get("max_backlinks", 7)
    print(f"현재 max_backlinks 설정: {max_backlinks}")
   
    # 1. 모든 노트의 임베딩 생성 (기존 임베딩 포함)
    notes_data = process_vault_embeddings(config)
   
    if not notes_data:
        print("처리된 노트가 없습니다. 프로그램을 종료합니다.")
        return
   
    # 2. 관련 노트 탐색 (전체 노트 데이터 사용)
    related_notes = find_related_notes(notes_data, config)
   
    if not related_notes:
        print("관련 노트가 없습니다. 프로그램을 종료합니다.")
        return
   
    # 3. 백링크 추가
    update_notes_with_backlinks(related_notes, config)
   
    # 종료 시간 기록 및 소요 시간 계산
    end_time = time.time()
    elapsed_time = end_time - start_time
   
    # 처리 결과 출력
    print(f"\n처리 완료: 총 {len(notes_data)}개 노트 임베딩 생성, {len(related_notes)}개 노트에 백링크 추가")
    print(f"총 소요 시간: {elapsed_time:.2f}초, 평균: {elapsed_time/max(len(notes_data), 1):.2f}초/노트")
   
    # 테스트 모드 안내
    if test_mode:
        print("\n테스트 모드로 실행되었습니다. 전체 노트를 처리하려면 config.json에서 'test_mode'를 false로 설정하세요.")

if __name__ == "__main__":
    main()