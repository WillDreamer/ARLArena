#!/usr/bin/env python3
import subprocess
import time
import sys

# =======================
# 配置区（你一般只改这里）
# =======================

GPU_ID = 4                      # 第 4 张卡（0-based）
MEM_THRESHOLD_MB = 1000          # 认为“空”的显存阈值
CHECK_INTERVAL = 30              # 每 30 秒检查一次
STABLE_CHECKS = 3                # 连续 3 次满足才触发
BASH_SCRIPT = "/data1/dannie/projects/ARLArena/examples/game_agent_trainer/train_gspo_rft.sh"  # ❗改成你的 bash 路径

# =======================
# GPU 查询函数
# =======================

def query_gpu_memory():
    """
    返回 dict: {gpu_id: used_memory_MB}
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd).decode().strip().split("\n")

    mem = {}
    for line in out:
        idx, used = line.split(",")
        mem[int(idx.strip())] = int(used.strip())
    return mem


def gpu_is_free(mem_info):
    """
    判断 GPU_ID 是否显存足够低
    """
    used = mem_info.get(GPU_ID, None)
    if used is None:
        return False
    return used < MEM_THRESHOLD_MB


# =======================
# 触发 bash
# =======================

def run_bash():
    print(f"\n🚀 Launching bash script: {BASH_SCRIPT}\n")
    subprocess.run(["bash", BASH_SCRIPT])


# =======================
# 主逻辑
# =======================

def main():
    stable_cnt = 0
    print(f"👀 Watching GPU {GPU_ID} ...")

    while True:
        try:
            mem = query_gpu_memory()
            used = mem.get(GPU_ID, None)

            print(
                f"[GPU {GPU_ID}] used={used} MiB | "
                f"stable={stable_cnt}/{STABLE_CHECKS}"
            )

            if gpu_is_free(mem):
                stable_cnt += 1
                print("🟢 Condition satisfied")
            else:
                if stable_cnt > 0:
                    print("🔄 Condition broken, reset counter")
                stable_cnt = 0

            if stable_cnt >= STABLE_CHECKS:
                print("\n✅ GPU is stably free!")
                run_bash()
                print("🛑 Done. Exiting watcher.")
                sys.exit(0)

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
