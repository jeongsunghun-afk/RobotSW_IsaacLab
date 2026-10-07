"""R.pet 인계 play 스크립트를 IsaacLab 2.3.2(Isaac Sim 5.1)에서 돌리는 런처 어댑터.

왜 필요한가
-----------
인계본은 IsaacLab 3.0.0 용이다. 3.0.0 은 ``lazy_export`` + ``.pyi`` 스텁 구조라
``isaaclab*`` 를 **앱 실행 전에** import 할 수 있고, 그래서 원본
``examples/play_in_isaaclab.py:53`` 이 최상단에서

    from isaaclab_tasks.utils import add_launcher_args, launch_simulation

를 한다. 우리 2.3.2 는 eager 라 그 import 가 즉시 ``pxr``(Isaac Sim USD 바인딩)을 요구하고
앱이 아직 없어 ``ModuleNotFoundError: No module named 'pxr'`` 로 죽는다.
게다가 그 헬퍼 두 개는 3.0.0 에서 추가된 것이라 2.3.2 에 **아예 없다**.

접근
----
원본을 수정하지 않는다. 대신 이 어댑터가
  1) ``isaaclab.app.AppLauncher`` 로 **앱을 먼저** 띄운다(2.x 표준 규약, 우리 Go2 play.py 와 동일).
  2) 앱이 뜨면 ``pxr`` 이 생기므로 ``isaaclab_tasks.utils`` import 가 성공한다.
     거기에 없는 ``add_launcher_args`` · ``launch_simulation`` 두 심볼만 2.x 등가로 **주입**한다.
  3) 원본 스크립트를 ``__main__`` 으로 실행한다.
이렇게 하면 원본의 모든 로직(env.yaml 반영·estimator 부착 검사·요약 통계)이 그대로 쓰인다.

쓰기
----
  python rpet_play_2x.py --orig <play_in_isaaclab.py 경로> [원본 인자들...]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys

# ── 1) 앱 먼저 (이 단계에서는 isaaclab.app 만 import 한다 — pxr 불필요) ──────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--orig", required=True, help="원본 play_in_isaaclab.py 절대경로")
_pre_args, _rest = _pre.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_launch_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_launch_parser)
_launch_args, _forward = _launch_parser.parse_known_args(_rest)

app_launcher = AppLauncher(_launch_args)
simulation_app = app_launcher.app

# ── 2) 이제 pxr 이 있다 → 누락 헬퍼 주입 ────────────────────────────────────────
import isaaclab_tasks.utils as _iu  # noqa: E402


def _add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """3.0.0 의 add_launcher_args 등가. AppLauncher 는 이미 떴으니 인자만 다시 등록한다
    (원본이 자기 parser 로 --headless 등을 파싱할 수 있게 해 주는 것이 목적)."""
    AppLauncher.add_app_launcher_args(parser)


@contextlib.contextmanager
def _launch_simulation(env_cfg, args):
    """3.0.0 의 launch_simulation 등가.

    3.0.0 판은 이 컨텍스트 안에서 앱을 띄우고 나올 때 닫는다. 우리는 앱을 이미 띄웠으므로
    **device 반영만 하고** 종료 시 닫는다. 원본이 요약 출력을 이 컨텍스트 **안에서** 하므로
    (원본 주석 :252) 닫는 시점을 바꾸면 요약이 안 찍힌다 — 그래서 finally 에서 닫는다.
    """
    dev = getattr(args, "device", None)
    if dev:
        env_cfg.sim.device = dev
    try:
        yield
    finally:
        simulation_app.close()


if not hasattr(_iu, "add_launcher_args"):
    _iu.add_launcher_args = _add_launcher_args
if not hasattr(_iu, "launch_simulation"):
    _iu.launch_simulation = _launch_simulation

# ── 3) 원본 실행 ──────────────────────────────────────────────────────────────
_orig = os.path.abspath(_pre_args.orig)
if not os.path.isfile(_orig):
    raise SystemExit(f"원본 스크립트를 찾을 수 없다: {_orig}")

# 원본이 자기 디렉터리의 모듈을 import 할 수 있게 (standalone/ 등)
sys.path.insert(0, os.path.dirname(_orig))
sys.argv = [_orig] + _forward
print(f"[adapter] IsaacLab 2.3.2 런처 어댑터 — 앱 선행 기동 후 원본 실행")
print(f"[adapter] orig  : {_orig}")
print(f"[adapter] argv  : {' '.join(_forward)}")
print(f"[adapter] 주입  : add_launcher_args, launch_simulation (2.3.2 에 없는 3.0.0 헬퍼)")
runpy.run_path(_orig, run_name="__main__")
