# Mlflow
**샘플데이터는 제공하지 않습니다.**<br>
**기본 파이썬 가상환경은 아나콘다 혹은 미니콘다로 구성하는것을 추천(pip로 오류걸리는 라이브러리들이 간혹 있음)**
**텐서플로우와 뉴럴프로펫을 합친 버전입니다**
------
## 0. 필수
1. Database
2. 리눅스 기반 sftp서버
3. Mlflow 기본

## 1. 시스템 요구사항
###  python: 3.8 or upper
- neuralprophet
- tensorflow
- SQLAlchemy 1.4.42
- mlflow 2.2.0
- pysftp
- psycopg2 -> conda 설치 추천
- hyperopt

### Database
- postgresql

### sftp Server
- CentOS7

## 2. 사용법
**setting.yaml**에서 원하는 항목을 수정, **user_modify.py**에서 상황에 맞게 전처리 코드를 입력하시면 됩니다.

뉴럴프로펫은 변수가 무조건 들어가도록 설정되어 있으며, Tensorflow는 예측 손실 함수만 입력되어있습니다.


MLFLOW_TRACKING_URI: MLFLOW tracking uri 입렵<br>
ARTIFACT_URI: Artifact uri 입력(sftp로 테스트해봄)<br>
data_path: 데이터 폴더 및 파일 입력

mlflow 내에 y_true(actual), y_pred(prediction)의 값도 입력되도록 제작되었습니다.

훈련이 끝나면 자동적으로 훈련된 모델을 인증하도록 하였습니다.

기타 수정사항은 직접해야합니다.

