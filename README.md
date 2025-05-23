# 옵시디언 노트 처리 시스템

이 시스템은 옵시디언(Obsidian) 마크다운 노트를 위한 자동화 도구로, 로컬 LLM을 활용하여 노트 내용에서 요약과 태그를 생성하고, 임베딩을 통해 관련 노트를 자동으로 연결합니다.

## 기능 개요

이 시스템은 다음 주요 기능을 제공합니다:

1. **노트 요약 생성**: 각 노트의 내용을 분석하여 간결한 요약을 자동으로 생성합니다.
2. **관련 태그 추출**: 노트 내용에서 핵심 키워드를 태그로 추출합니다.
3. **벡터 임베딩 생성**: 노트의 의미를 고차원 벡터 공간에 매핑합니다.
4. **관련 노트 연결**: 유사도 계산을 통해 관련 노트를 찾고 백링크로 연결합니다.

## 사전 요구사항

* **Python 3.7 이상**
* **Ollama** - 로컬 LLM 실행을 위한 도구 (https://ollama.ai)
* **설치된 모델**:
  * gemma3:12b - 요약 및 태그 생성용
  * mxbai-embed-large - 임베딩 생성용
* **Python 패키지**: pyyaml, numpy, scipy, requests

## 설치 방법

1. 이 저장소를 로컬 컴퓨터에 클론합니다:
```
git clone https://github.com/yourusername/obsidian-processor.git
cd obsidian-processor
```

2. 필요한 Python 패키지를 설치합니다:
```
pip install pyyaml numpy scipy requests
```

3. Ollama를 설치하고 필요한 모델을 다운로드합니다:
```
ollama pull gemma3:12b
ollama pull mxbai-embed-large
```

4. `config.json` 파일을 자신의 환경에 맞게 수정합니다. (아래 설정 부분 참조)

## 사용 방법

### 기본 사용법

1. Ollama 서버가 실행 중인지 확인합니다 (기본 포트: 11434).

2. 메인 처리 스크립트를 실행합니다:
```
python obsidian_processor.py
```

3. 처리 모드를 선택합니다:
   - **전체 재처리**: 모든 노트의 요약과 임베딩을 다시 생성합니다.
   - **증분 요약 처리**: 새 노트만 요약을 생성하고, 전체 임베딩을 처리합니다.
   - **요약만 처리**: 새 노트만 요약을 생성합니다.
   - **전체 임베딩 처리**: 임베딩 생성 및 모든 백링크를 재계산합니다.

4. 처리가 완료되면 결과가 콘솔에 표시되고 로그 파일에 기록됩니다.

### 개별 스크립트 실행

각 스크립트를 개별적으로 실행할 수도 있습니다:

* 요약 및 태그 생성:
```
python obsidian_summary.py
```

* 임베딩 및 백링크 생성:
```
python obsidian_embedding.py
```

## 설정 (config.json)

```json
{
    "model": "gemma3:12b",            // 요약 생성에 사용할 모델
    "embedding_model": "mxbai-embed-large",  // 임베딩 생성에 사용할 모델
    "vault_paths": [                  // 처리할 옵시디언 볼트 경로들
        "D:\\z-vault\\03. Knowledge",
        "D:\\z-vault\\01. Think & Plan",
        "D:\\z-vault\\02. Projects",
        "D:\\z-vault\\00. News"
    ],
    "skip_categories": ["knowledge", "plan", "main-project"],  // 처리에서 제외할 카테고리
    "exclude_directories": [          // 처리에서 제외할 디렉토리
        "D:\\z-vault\\00. News\\original"
    ],
    "min_content_length": {           // 최소 컨텐츠 길이 설정
        "summary": 50,                // 요약 생성을 위한 최소 길이
        "embedding": 100              // 임베딩 생성을 위한 최소 길이
    },
    "summary_length": {               // 요약 길이 설정
        "short": 1,                   // 짧은 노트 요약 줄 수
        "medium": 2,                  // 중간 노트 요약 줄 수
        "long": 4                     // 긴 노트 요약 줄 수
    },
    "tag_count": {                    // 태그 개수 설정
        "min": 3,                     // 최소 태그 개수
        "max": 5                      // 최대 태그 개수
    },
    "metadata_fields": [              // 임베딩에 포함할 메타데이터 필드
        "title", "category", "summary", "tags", "source", "date"
    ],
    "embedding": {                    // 임베딩 관련 설정
        "similarity_threshold": 0.7,  // 관련 노트로 간주할 최소 유사도
        "max_backlinks": 3,           // 각 노트당 최대 백링크 수
        "jsonl_path": "D:\\z-vault\\.vectors\\embeddings.jsonl"  // 임베딩 저장 경로
    },
    "api_settings": {                 // API 호출 설정
        "temperature": 0.1,           // 생성 모델의 temperature
        "top_p": 0.9,                 // 생성 모델의 top_p
        "max_tokens": 500             // 생성 모델의 최대 토큰 수
    },
    "parallel_processing": true,      // 병렬 처리 활성화 여부
    "num_workers": 8,                 // 병렬 처리 시 워커 수
    "test_mode": false,               // 테스트 모드 활성화 여부
    "force_reprocess": false          // 강제 재처리 여부
}
```

## 출력 결과

### 프론트매터 형식

처리 후 각 노트의 프론트매터에는 다음 항목이 추가됩니다:

```yaml
---
title: '노트 제목'
category: '카테고리'
summary: '자동으로 생성된 노트 요약 내용'
tags:
  - '태그1'
  - '태그2'
  - '태그3'
processed:
  - 'summary # 2025-05-05'
  - 'embedding # 2025-05-05'
  - 'backlinks # 2025-05-05'
related_notes:
  - "[[관련 노트 1]]" # similarity: 0.92
  - "[[관련 노트 2]]" # similarity: 0.85
  - "[[관련 노트 3]]" # similarity: 0.79
---
```

### 로그 파일

처리 과정은 `obsidian_process.log` 파일에 기록되며, 다음과 같은 정보가 포함됩니다:

```
[2025-05-05 12:34:56] ===== 옵시디언 노트 처리 시작 (2025-05-05 12:34:56) =====
[2025-05-05 12:34:58] 선택한 모드: 증분 요약 처리 (새 노트만 요약 + 전체 임베딩 처리)
[2025-05-05 12:35:30] 1. 노트 요약 및 태그 생성: 15개 노트 처리 완료, 0개 오류, 소요 시간: 32.45초
[2025-05-05 12:36:45] 2. 노트 임베딩 및 백링크 생성: 15개 노트 임베딩 생성, 120개 노트에 백링크 추가, 소요 시간: 75.12초
[2025-05-05 12:36:45] ===== 처리 완료 =====
```

## 처리 모드 설명

### 1. 전체 재처리

이 모드는 모든 노트를 처음부터 다시 처리합니다:
- 모든 노트의 요약과 태그를 다시 생성
- 모든 노트의 임베딩을 다시 생성
- 모든 노트 쌍의 유사도를 재계산
- 모든 노트의 백링크를 업데이트

처음 설정하거나 중요한 변경이 있을 때 사용합니다.

### 2. 증분 요약 처리

이 모드는 효율적인 증분 처리를 제공합니다:
- 새로운/변경된 노트만 요약과 태그 생성
- 새로운/변경된 노트만 임베딩 생성
- 하지만 모든 노트 쌍의 유사도는 재계산
- 모든 노트의 백링크 업데이트

일상적인 사용에 가장 적합한 모드입니다.

### 3. 요약만 처리

요약 기능만 필요할 때 사용합니다:
- 새로운/변경된 노트만 요약과 태그 생성
- 임베딩 및 백링크 작업은 수행하지 않음

### 4. 전체 임베딩 처리

임베딩과 백링크만 업데이트할 때 사용합니다:
- 요약 작업은 수행하지 않음
- 새로운/변경된 노트만 임베딩 생성
- 모든 노트 쌍의 유사도 재계산
- 모든 노트의 백링크 업데이트

백링크 설정을 변경한 경우 유용합니다.

## 참고 사항

1. **Ollama 서버**: 스크립트 실행 전 Ollama 서버가 실행 중이어야 합니다.
2. **처리 시간**: 노트 수에 따라 처리 시간이 길어질 수 있습니다.
3. **백업**: 처음 사용 시 볼트 백업을 권장합니다.
4. **테스트 모드**: 새로운 볼트에서는 먼저 `test_mode: true`로 설정하여 테스트하세요.
5. **유사도 임계값**: `similarity_threshold` 값을 조정하여 관련 노트의 품질을 제어할 수 있습니다.

## 문제 해결

* **Ollama 연결 오류**: Ollama 서버가 실행 중인지 확인하세요 (`http://localhost:11434`).
* **임베딩 오류**: 임베딩 모델이 올바르게 설치되었는지 확인하세요.
* **YAML 파싱 오류**: 프론트매터에 이스케이프되지 않은 특수 문자가 있는지 확인하세요.
* **처리가 너무 느림**: `parallel_processing`를 활성화하고 `num_workers`를 조정하여 속도를 개선할 수 있습니다.

## 개발자 정보

이 시스템은 옵시디언 노트의 관리와 연결을 개선하기 위해 개발되었습니다. 버그 신고나 개선 제안은 이슈 트래커를 이용해주세요.

## 라이선스

MIT 라이선스를 따릅니다.