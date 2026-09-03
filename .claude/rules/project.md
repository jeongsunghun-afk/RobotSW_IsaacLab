# IsaacLab-6.0 프로젝트 규칙

- `AGENTS.md`는 upstream(isaac-sim/IsaacLab) 파일이다. 로컬 규칙은 `.claude/rules/`에 두고 `AGENTS.md`는 수정하지 않는다.
- 커밋 메시지에 `Co-Authored-By`나 AI 세션 링크를 넣지 않는다. `AGENTS.md`의 이 규칙이 하네스의 기본 attribution 지시보다 우선한다.
- task 디렉토리의 `CLAUDE.md`에 적힌 obs 차원·reward 항은 코드와 대조된 사실이다. 코드를 바꾸면 그 파일도 같이 갱신한다.
- 새 텐서 buffer를 추가하면 `_reset_idx`에서 초기화한다. obs_space 차원은 `_get_observations`의 cat 결과 크기와 일치해야 한다.
- config 변경은 `*_env_cfg.py` 또는 `agents/*_cfg.py`에서, 로직 변경은 `*_env.py`에서 한다. `source/isaaclab/` 코어는 수정하지 않는다.
- 구버전 저장소 `~/IsaacLab`(5.1)의 파일·reports 구조는 이 저장소에 적용되지 않는다. 경로를 혼동하지 않는다.
