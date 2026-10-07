import io, ast
p=("/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6/source/isaaclab_tasks/isaaclab_tasks/"
   "direct/leg_imitation_tracking/leg_imitation_tracking_env.py")
s=io.open(p,encoding="utf-8").read()

old = """        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
"""
assert s.count(old)==1, s.count(old)

new = '''        # [JSH 2026-10-06] RPET_TERRAIN 게이트로 험지를 투입한다. 미설정 시 아래 평지 경로가
        # 그대로 돌아 **동작이 바이트 동일**하다(zero-shot 기준선 보존).
        #   왜 cfg 스위치가 아닌가: cfg 의 `terrain: TerrainImporterCfg` 필드는 **아무도 쓰지 않는
        #   죽은 설정**이고(이 함수가 spawn_ground_plane 을 직접 부른다) terrain_generator=None 이다.
        #   값: rough(완만 범프) | gap(희소 갭) | stepping(징검돌).  RPET_TERRAIN_LEVEL=0..9 로 난도.
        _rpet_terrain = os.environ.get("RPET_TERRAIN", "").strip().lower()
        if _rpet_terrain:
            import isaaclab.terrains as terrain_gen
            from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
            from isaaclab.terrains import TerrainImporter

            _lvl = int(os.environ.get("RPET_TERRAIN_LEVEL", "5"))
            _f = max(0.0, min(1.0, _lvl / 9.0))   # 0=쉬움 … 1=어려움
            if _rpet_terrain == "rough":
                _sub = {"r": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0, noise_range=(0.0, 0.02 + 0.08 * _f), noise_step=0.02,
                    border_width=0.25)}
            elif _rpet_terrain == "gap":
                _sub = {"g": terrain_gen.MeshGapTerrainCfg(
                    proportion=1.0, gap_width_range=(0.1 + 0.3 * _f, 0.15 + 0.35 * _f),
                    platform_width=2.0)}
            elif _rpet_terrain == "stepping":
                _sub = {"s": terrain_gen.HfSteppingStonesTerrainCfg(
                    proportion=1.0,
                    stone_height_max=0.0 + 0.05 * _f,
                    stone_width_range=(0.40 - 0.28 * _f, 0.45 - 0.30 * _f),
                    stone_distance_range=(0.02 + 0.16 * _f, 0.05 + 0.18 * _f),
                    holes_depth=-0.5, platform_width=1.5)}
            else:
                raise ValueError(f"RPET_TERRAIN={_rpet_terrain!r} 는 rough|gap|stepping 중 하나여야 한다")

            _gen = TerrainGeneratorCfg(
                size=(8.0, 8.0), border_width=20.0, num_rows=4, num_cols=4,
                horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
                use_cache=False, sub_terrains=_sub,
            )
            self._terrain = TerrainImporter(TerrainImporterCfg(
                prim_path="/World/ground", terrain_type="generator", terrain_generator=_gen,
                max_init_terrain_level=None, collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                debug_vis=False,
            ))
            print(f"[RPET-TERRAIN] kind={_rpet_terrain} level={_lvl} (f={_f:.2f}) — "
                  f"평지 대신 생성지형 투입. 종료/스폰 z 는 지형상대 보정 필요(별도 게이트).")
        else:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )
'''
s=s.replace(old,new,1)
assert "import os" in s or "\nimport os\n" in s, "os 미임포트"
ast.parse(s)
io.open(p,"w",encoding="utf-8").write(s); print("RPET_TERRAIN 게이트 추가 (미설정 시 동작 불변)")
