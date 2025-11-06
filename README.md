# open-mission

## 📌 1. 미션 선택 

이번 우테코 오픈 미션의 핵심은
**“스스로 도전하고 싶은 문제를 정의하고, 이를 해결하기 위해 새로운 기술을 직접 탐색하며 프로젝트를 완성하는 것”**이었다.

나는 현재 인턴십에서 10만 장 이상의 이미지로 YOLO 기반 화재·연기 감지 모델을 직접 학습하며,
모델 개발과 데이터 엔지니어링에 깊게 몰입해왔다.
하지만 모델의 성능을 확인할 때마다 항상 Python 스크립트로만 추론을 확인해야 했다.

이 방식은

- 이미지/영상 테스트가 불편하고,
- 다른 사람에게 설명하기 어렵고,
- 서비스 형태로 확장하기 어렵다는 한계
가 있었다.

그래서 이번 미션에서는 단순히 “모델을 실행하는 코드”를 넘어서,
`내가 직접 학습한 AI 모델을 실제 ‘웹 서비스 형태’로 제공해보고 싶다는 목표를 세웠다.`

사용자가 직접 이미지를 넣으면
→ 서버에서 모델이 추론하고
→ 그 결과를 시각적으로 확인할 수 있는 형태의 서비스.

단순 스크립트가 아닌, 완전한 end-to-end AI 추론 서비스를 구현하는 것이 이번 프로젝트의 핵심 동기이자 도전 목표였다.

---

## 📌 2. 프로젝트 개요

이 프로젝트는 YOLOv5 객체 탐지 모델을 FastAPI 기반 웹서비스로 제공하는 것을 목표로 한다.

사용자는 웹 페이지에서 이미지 또는 동영상을 업로드할 수 있고,
서버는 업로드된 데이터를 YOLOv5 모델로 분석해

- 객체 위치(Bounding Box)
- 클래스 라벨
- 신뢰도(Confidence Score)
를 JSON 형태로 반환한다.

프론트엔드는 이 결과를 받아
Canvas 기반 렌더링을 통해 탐지된 객체를 이미지 위에 시각적으로 표시한다.

### 🎈 전체 흐름

[사용자] → (이미지/동영상 업로드) → [FastAPI 서버] → (YOLO 추론)
→ (JSON으로 결과 반환) → [프론트엔드 시각화]

### 🎈 제공 기능
- 이미지/동영상 파일 업로드
- YOLOv5 모델을 활용한 객체 탐지(Inference)
- bounding box + label 시각화
- 신뢰도 표시
- 추론 
- 비동기 기반 FastAPI 서버
- 최근 추론 결과 저장(이미지·동영상 History 기능)

### 💡 FastAPI를 선택한 이유

이번 프로젝트의 가장 중요한 목표는
**“AI 모델이 실제 서비스로 동작하는 흐름을 구현하는 것”**이었다.

이 목표에서 FastAPI는 다음과 같은 강점이 있다고 판단하였다:

1) AI·딥러닝 생태계와의 완벽한 궁합

FastAPI는 Python 기반이므로
- PyTorch
- OpenCV
- Numpy
- YOLO inference 코드
와 자연스럽게 엮인다.
모델 로딩, 전처리, 후처리 등을 별도의 변환 작업 없이 바로 구현할 수 있다.

반면 Java(Spring)으로 구현할 경우:
- 모델을 ONNX로 변환해야 하고
- ByteBuffer 기반 전처리
- Custom 후처리(CNN output 직접 파싱)
- Java와 딥러닝 생태계 간 간극 해소

같은 ‘기술적 부수 작업’이 훨씬 많아진다.
이것은 프로젝트의 초점을 “AI 서비스 구축”에서 “자바 생태계에 모델을 우겨 넣기”로 흐트러뜨린다.

2) 파일 업로드, JSON 응답, 비동기 처리 등 웹 서비스 구축에 최적화

FastAPI는
- 파일 업로드 처리
- JSON 직렬화
- CORS 설정
- 비동기 서버
- 라우팅 분리
같은 웹 개발 기능을 매우 직관적으로 제공한다.
따라서 AI inference API 구조를 설계하기 쉬워,
모델 추론 로직에 집중할 수 있다.

3) AI 모델 서빙 표준으로 자리 잡은 기술

FastAPI는 속도, 단순함, 문서화 자동화 덕분에
많은 기업에서 실제로 ML 모델 서빙용 프레임워크로 쓰인다.

`즉, FastAPI는 “AI 모델을 서비스로 만든다”는 이번 프로젝트의 목적에 가장 적합한 기술 스택이었다.`

---

## 📌 3. 기술 스택

### Backend
- FastAPI
- Uvicorn
- Python
- YOLOv5 (PyTorch)
- OpenCV
- Numpy

### Frontend
- HTML5
- Vanilla JavaScript

### Infra / Architecture
- REST API 기반 구조
- 이미지/영상 파일 처리
- 모델 메모리 상주 로딩(Warm-up)
- 비동기 기반 FastAPI 서버
- 모듈 기반 설계 (Router/Service/Utils)

---

## 📌 4. 2주 프로젝트 계획 (기능 단위 Commit 기반)

아래 계획은 FastAPI 기반 YOLO 추론 웹 서비스를 2주 동안 단계별로 구축하기 위한 개발 흐름을 정리한 것이다.
각 단계는 구현해야 할 기능과 관련된 commit 단위로 구성되어 있어, 명확하고 일관성 있게 개발을 진행할 수 있다.

1️⃣ 프로젝트 초기 세팅 & 구조 설계 

본격적인 개발 전에 전체 서비스 구조를 설계하고, FastAPI 환경을 구축한다.
- [docs] README 초안 작성 (기획·목표·구조 정리)
- [feat] 프로젝트 초기 폴더 구조 생성
- [feat] FastAPI 기본 서버(main.py) 초기화
- [feat] API 라우팅 구조(api/v1) 설계


2️ YOLO 모델 로딩 & 추론 환경 구축 

YOLO 모델을 FastAPI 서버에 안정적으로 로드할 수 있는 기반을 마련한다.
- [feat] ModelLoader 모듈 구현 (YOLO 모델 로딩)
- [feat] 모델 warm-up 기능 추가
- [refactor] 모델 로딩 실패 대비 예외 처리


3️⃣ 이미지/동영상 입력 처리 

FastAPI에서 업로드된 이미지·동영상 파일을 처리하고,
YOLO 모델로 전달할 수 있는 형태(ndarray)로 변환한다.
- [feat] 이미지 입력 API 기본 구조 구현
- [feat] 이미지 UploadFile → numpy ndarray 변환 로직 추가
- [feat] 전처리 모듈 이미지 처리 기능 구현
- [feat] 비디오 입력 API 기본 구조 구현
- [feat] 동영상 대표 프레임 추출 기능 구현
- [refactor] 서비스 레이어와 라우터 구조 분리


4️⃣ YOLO Inference 서비스 구축 

YOLO 모델로 이미지/영상을 추론하고, 결과를 표준화된 JSON 포맷으로 변환한다.
- [feat] YOLO 추론 엔진 구현
- [feat] Bounding Box + Label + Confidence 계산 로직 작성
- [feat] 표준화된 JSON 응답 포맷 설계
- [refactor] 라우터–서비스 계층 분리 및 구조 개선


5️⃣ 프론트엔드 UI 구축 

웹 페이지에서 사용자가 직접 이미지를 업로드하고,
API 결과를 확인할 수 있는 기본 UI를 구성한다.
- [feat] index.html UI 구성
- [feat] 업로드 버튼 + 파일 선택 UI 구현
- [style] 기본 스타일링(CSS) 추가
- [feat] fetch API로 /predict/image 연동
- [feat] fetch API로 /predict/video 연동
- [feat] 응답 JSON 파싱 로직 작성


6️⃣ Canvas 기반 Bounding Box 시각화 

YOLO 추론 결과를 직관적으로 확인할 수 있도록 Canvas 기반 시각화 기능을 구현한다.
- [feat] Canvas 기반 bounding box 렌더링 구현
- [feat] Label + Confidence 텍스트 표시 기능
- [refactor] 렌더링 로직 별도 JS 모듈로 분리
- [style] UI 개선 (색상, 테두리, 폰트 등)


7️⃣ 최근 추론 결과 저장(Result History) 기능

사용자가 업로드한 **원본 이미지·추론 결과 이미지·메타데이터(JSON)**를 저장하고,
이력을 조회할 수 있는 기능을 추가한다.
- [feat] ResultStorageService 구현
- [feat] 추론 결과 이미지 저장(result.png)
- [feat] 원본 이미지 저장(original.png)
- [feat] 추론 메타데이터(JSON) 저장
- [feat] timestamp 기반 파일명 생성
- [feat] 최근 N개 결과 조회 기능 구현
- [feat] 상세 결과 조회 기능 구현
- [feat] 정적 파일 제공(StaticFiles) 설정


8️⃣ History UI & 사용자 경험 개선

저장된 추론 이력을 사용자가 쉽게 확인할 수 있도록 History UI를 확장·개선한다.

- [feat] 최근 추론 결과 썸네일 리스트 UI 구현
- [feat] 상세 보기 모달/뷰어 화면 구현
- [style] 반응형 및 UX 중심 UI 개선
- [refactor] 프론트 코드 구조 개선


9️⃣ 통합 테스트 & 문서화 & 최종 정리 (Finalization)

전체 시스템을 통합하고, 문서화 및 마무리 작업을 진행한다.

- [test] end-to-end 테스트
- [refactor] 전체 코드 구조 정리
- [docs] README 최종 작성 (개요·구조·학습 내용·실행 방법)
- [docs] 아키텍처/흐름도 다이어그램 추가


## 📌 5. 폴더 구조


