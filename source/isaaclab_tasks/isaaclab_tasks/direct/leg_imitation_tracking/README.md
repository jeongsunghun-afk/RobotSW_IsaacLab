# R.pet Phase-T 학습 데이터셋 (s16, 2026-07-28)

s16(Phase-T) AMP 학습이 사용 중인 16클립. dataset_rpet_walktrot_T.yaml 그대로.

## 구성
- **walk 밴드 (hrfix 수술본, 8클립)**: walk050f/065f/080f (0.50/0.65/0.79 m/s, walk065 시간 리샘플)
  + ramp58f (0.49→0.76 m/s 가속 램프) + 각 미러(fm). 원본 소스: DogML D1_ex02_061 계열 리타게팅.
  hrfix = HR_foot_joint 축 반전(0,-1,0) 미정합으로 인한 우측 뒷발목 41.6° 과굴곡·지면관통 수술본
  (대칭 발목 위상이식 + thigh/calf/foot 3관절 xz IK). 검수: 4발 min z 2.3~2.5cm(발끝 구 반경 정합),
  뒤 듀티Δ 4~6%, 뒤 들림Δ 0~3%.
- **trot 밴드 (8클립)**: trot090/110/130/150 (0.90/1.10/1.29/1.49 m/s) + 미러(m).
  원본 소스: DogML r009b 리타게팅 → s12 수술 계보(재접지·hip상대 보폭압축 K·앞다리 거버너·가속리미터,
  밴드별 시간 리샘플). 07-28 감사: HR 결함 없음(뒤 들림Δ 5~6%, 발목 대칭), 캡초과 1%대.

## 포맷
pkl: {loop_mode: 0(CLAMP), fps: 60, frames: [T, 23]}
frames 열: [0:3] root xyz, [3:6] root 회전 axis-angle(expmap), [6:23] 관절각 17개(rad).
- rpet_xmlorder/: 관절 순서 = MuJoCo XML [HL_hip,HL_thigh,HL_calf,HL_foot, HR_*, waist, FL_*, FR_*]
- rpet_usd_order/: 관절 순서 = Isaac USD [waist, FL_hip..FL_foot, FR_*, HL_*, HR_*] (학습 로더용)
⚠ numpy 1.26 직렬화 (2.x로 재저장 시 학습 파이썬에서 로드 실패 주의)

## 알려진 한계
- walk 밴드는 단일 클립 리샘플이라 보폭 83cm 고정(속도-보폭 관계 부재), root 롤 p-p ~0°(리타게터 아티팩트).
- trot 밴드 접지 슬립 11~18cm/s 잔존(스탠스 고정 수술 미적용).
