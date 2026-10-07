import io
p = "/mnt/ssd1/jsh/rs_table_arms.py"
s = io.open(p, encoding="utf-8").read()
old = "SEEDS = [42, 7, 1, 3, 2, 5, 11, 23]"
assert s.count(old) == 1, s.count(old)
new = ('SEEDS = [int(x) for x in os.environ.get("RS_SEEDS", "42,7,1,3,2,5,11,23").split(",")]\n'
       '# ↑ 시드 집합을 RS_SEEDS 로 인자화.  고정 [42,7,1,3,2,5,11,23] 는 d-계열 정책의 규약이고,\n'
       '#   팔 비교(armG/armT/armTS)는 1~8 로 돌렸다.  교집합만 세면 n=5 로 줄어 8-seed 규약이 깨진다.')
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("SEEDS 인자화 OK (RS_SEEDS)")
