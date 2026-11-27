# 데이터 사전 (30개 사용 변수)

UCI ML Repository 데이터셋 697 *Predict Students' Dropout and Academic Success*에서 실제로 사용한 30개 열을 정리한 문서입니다. 원본은 4,424행 36개 설명변수이고 결측 세포는 0개입니다. 2학기 교과 이수 열 6개(`curricular_units_2nd_sem_*`)는 예측 시점 이후에 기록되므로 전부 제외했습니다. 열 이름은 `normalise_columns`를 거친 뒤의 이름입니다.

값 형식에서 "관측"이라고 적은 범위는 원본 파일 전체 4,424행에서 확인한 최솟값과 최댓값이며, 데이터의 이론적 정의역이 아닙니다.

| 열 이름 | 뜻 | 단위 또는 값 형식 | 이 문제에서 왜 보는가 |
|---|---|---|---|
| marital_status | 결혼 상태 | 코드값 (UCI 메타데이터 참조) | 학업 외 생활 조건이 이탈 위험과 어떻게 얽혀 있는지 보기 위해 둡니다. 코드의 실제 의미는 메타데이터로 확인해야 합니다. |
| application_mode | 입학 전형(지원 방식) | 코드값 (UCI 메타데이터 참조) | 어떤 경로로 들어온 학생인지가 초기 적응과 이탈 가능성에 대한 정보를 담을 수 있어 둡니다. |
| application_order | 지원 지망 순서 | 정수 코드값 (UCI 메타데이터 참조) | 지망 순서는 해당 과정이 학생의 1지망이었는지를 시사하며, 기대와 실제의 격차가 이탈로 이어질 수 있다는 가설을 검증하기 위해 둡니다. |
| course | 지원 학과(과정) | 코드값 (UCI 메타데이터 참조) | 학과별 교육과정 난이도와 이탈률 차이를 흡수하기 위해 둡니다. 코드 대응표는 메타데이터를 참조해야 합니다. |
| daytime_evening_attendance | 주간/야간 과정 구분 | 0 또는 1 (이진) | 주간과 야간은 재학 형태와 생활 여건이 달라 이탈 위험의 기준선이 다를 수 있어 둡니다. |
| previous_qualification | 이전 학력 자격 | 코드값 (UCI 메타데이터 참조) | 어떤 자격으로 입학했는지가 대학 과정 적응의 출발점을 설명할 수 있어 둡니다. |
| previous_qualification_grade | 이전 학력 성적 | 점 (관측 95.0~190.0) | 입학 전 학업 수준을 나타내는 지표로, 1학기 성적과 함께 읽어야 의미가 분명해집니다. |
| nacionality | 국적 | 코드값 (UCI 메타데이터 참조) | 국적은 언어·행정·생활 적응과 연결될 수 있는 배경 변수여서 둡니다. 코드 의미는 메타데이터를 참조해야 합니다. |
| mothers_qualification | 모(母)의 학력 | 코드값 (UCI 메타데이터 참조) | 가정의 교육적 배경이 학업 지속에 주는 영향을 보기 위해 둡니다. |
| fathers_qualification | 부(父)의 학력 | 코드값 (UCI 메타데이터 참조) | 모의 학력과 짝을 이루어 가정 배경을 이중으로 측정합니다. |
| mothers_occupation | 모(母)의 직업 | 코드값 (UCI 메타데이터 참조) | 가구의 사회경제적 위치를 간접적으로 반영하는 변수로 둡니다. |
| fathers_occupation | 부(父)의 직업 | 코드값 (UCI 메타데이터 참조) | 모의 직업과 함께 가구 조건을 구성하는 두 번째 축입니다. |
| admission_grade | 입학 성적 | 점 (관측 95.0~190.0) | 입학 시점의 학업 준비도를 나타내며, 1학기 성적과의 격차가 적응 실패의 신호가 될 수 있어 둡니다. |
| displaced | 거주지 이전(타지 출신) 여부 | 0 또는 1 (이진) | 타지에서 온 학생은 주거·생활비·사회적 지지에서 불리할 수 있다는 가설을 검증하기 위해 둡니다. |
| educational_special_needs | 특수교육 지원 대상 여부 | 0 또는 1 (이진) | 지원 필요 여부는 개입의 종류와 강도를 정하는 데 직접 관련되므로 둡니다. |
| debtor | 등록금 채무 보유 여부 | 0 또는 1 (이진) | 경제적 압박은 중도이탈의 대표적 경로이므로 둡니다. |
| tuition_fees_up_to_date | 등록금 납부 상태 | 0 또는 1 (이진) | 납부가 밀린 상태는 즉시 개입이 필요한 신호이며, 사람 규칙의 두 번째 조건이기도 합니다. |
| gender | 성별 | 0 또는 1 (이진) | 집단별 위험 양상이 다를 수 있어 보정 변수로 둡니다. 단독 해석은 피해야 합니다. |
| scholarship_holder | 장학금 수혜 여부 | 0 또는 1 (이진) | 장학금은 경제적 여유와 제도적 지지를 동시에 뜻하므로 둡니다. |
| age_at_enrollment | 입학 당시 나이 | 세 (관측 17~70) | 성인 학습자는 일과 학업을 병행하므로 이탈 위험의 구조가 다를 수 있어 둡니다. |
| international | 유학생 여부 | 0 또는 1 (이진) | 유학생은 행정·언어·체류 문제가 겹치므로 별도 신호로 둡니다. |
| curricular_units_1st_sem_credited | 1학기 인정(면제) 과목 수 | 과목 수, 정수 (관측 0~20) | 이미 인정받은 과목이 많으면 실제로 새로 이수해야 할 부담이 줄어드는 점을 반영합니다. |
| curricular_units_1st_sem_enrolled | 1학기 수강 신청 과목 수 | 과목 수, 정수 (관측 0~26) | 신청 규모 자체가 학업 참여의 강도를 보여 주며, 이수 과목 수와의 비율이 중요해집니다. |
| curricular_units_1st_sem_evaluations | 1학기 평가 응시 횟수 | 횟수, 정수 (관측 0~45) | 평가에 실제로 응시했는지는 수업 참여의 직접 증거이므로 둡니다. |
| curricular_units_1st_sem_approved | 1학기 이수 과목 수 | 과목 수, 정수 (관측 0~26) | 1학기 성과의 가장 직접적인 지표이며, 사람 규칙의 첫 번째 조건입니다. |
| curricular_units_1st_sem_grade | 1학기 평점 | 0~20 척도의 점수 (관측 0.0~18.875) | 이수 과목 수가 같아도 성적 분포가 다를 수 있어 질적인 성과를 보완합니다. |
| curricular_units_1st_sem_without_evaluations | 1학기 평가 없이 끝난 과목 수 | 과목 수, 정수 (관측 0~12) | 수강 신청은 했으나 평가에 이르지 못한 과목은 중도 포기의 초기 신호로 읽힙니다. |
| unemployment_rate | 실업률 | 퍼센트 (관측 7.6~16.2) | 졸업 후 노동시장 여건이 학업 지속의 기대 수익에 영향을 준다는 가설을 반영합니다. |
| inflation_rate | 물가상승률 | 퍼센트 (관측 -0.8~3.7) | 물가는 등록금과 생활비 부담을 통해 이탈 위험에 간접적으로 작용할 수 있어 둡니다. |
| gdp | 국내총생산 지표 | 실수 (관측 -4.06~3.51) | 거시 경제 상황을 나타내는 세 번째 지표로, 위 두 변수와 함께 경기 국면을 요약합니다. |

## 범주 코드에 대한 주의

`marital_status`, `application_mode`, `course`, `previous_qualification`, `nacionality`, `mothers_qualification`, `fathers_qualification`, `mothers_occupation`, `fathers_occupation`의 값은 정수 코드이며, 각 숫자가 무엇을 뜻하는지는 이 저장소 안에 기록되어 있지 않습니다. 본 문서는 코드 의미를 추정해 적지 않았습니다. 해석이 필요한 경우 UCI 데이터셋 페이지의 메타데이터(변수 설명과 값 대응표)를 직접 확인한 뒤 별도 표로 정리하십시오.

`displaced`, `educational_special_needs`, `debtor`, `tuition_fees_up_to_date`, `gender`, `scholarship_holder`, `international`, `daytime_evening_attendance`는 값이 두 개뿐인 이진 변수이며, 원-핫 인코딩 결과도 각각 두 열로 확인되었습니다.

## 사용하지 않은 열

2학기 교과 이수 6개 열(`curricular_units_2nd_sem_credited`, `curricular_units_2nd_sem_enrolled`, `curricular_units_2nd_sem_evaluations`, `curricular_units_2nd_sem_approved`, `curricular_units_2nd_sem_grade`, `curricular_units_2nd_sem_without_evaluations`)은 2학기 종료 후에 기록되므로 예측 시점 기준으로 누출에 해당하여 제외했습니다. 정답 열(`Target`)과 이를 이진으로 변환한 `target`은 레이블이므로 설계 행렬에 들어가지 않습니다.
