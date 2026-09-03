---
paths:
  - "reports/**"
  - "scripts/tools/report_video.py"
  - "_workspace/**"
---
# 실험 산출물 저장 규칙

- 영상·플롯·수치·비교 분석은 `reports/{task}/{experiment_name}/{run_dir}/`에 둔다. `_workspace/`나 저장소 루트에 만들지 않는다.
- `{task}`는 연구 주제 폴더(사람이 명명), `{experiment_name}`은 `agent_cfg.experiment_name`, `{run_dir}`는 `logs/rsl_rl/{experiment_name}/` 아래 디렉토리명이다. `rl_library` 레벨은 없다.
- 여러 run 비교는 `reports/{task}/_comparisons/{slug}/` 한 곳에만 둔다. 최상위 `reports/_comparisons/`는 폐지됐다.
- 새 experiment를 만들면 `reports/README.md`의 매핑 표와 `scripts/tools/report_video.py`의 `EXPERIMENT_TASK_MAP`을 둘 다 갱신한다.
- README에는 실측값과 params yaml에서 읽은 사실만 적는다. 모르면 `미확인`으로 남긴다.
- `.gitignore`는 `reports/` 아래 `.md`만 추적한다. mp4·png·csv가 커밋에서 빠지는 것은 의도된 동작이다.
- `logs/`는 원본, `reports/`는 사람이 다시 볼 것만 모은 큐레이션이다. mp4를 미러링하지 않고 README가 원본 경로를 역링크한다.
- `/report` 스킬(Notion 세션 보고서)과 별개다. 세션 보고서는 `reports/` 파일을 가리키고 결과물을 복제하지 않는다.

규약 전문: `reports/README.md`.
