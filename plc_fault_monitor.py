"""
plc_fault_monitor.py
====================
实时轴承故障诊断监控器

将 PLC 采集的振动信号实时送入训练好的 1D-CNN 模型进行故障诊断，
并将诊断结果以表格形式打印，同时（可选）写回 PLC。

运行方式
--------
  # 仿真模式（无需真实 PLC）
  python plc_fault_monitor.py

  # 指定故障类型仿真
  python plc_fault_monitor.py --fault inner --cycles 10

  # 连接真实 PLC（需安装 python-snap7）
  python plc_fault_monitor.py --plc-ip 192.168.0.1 --slot 1

依赖
----
  pip install tensorflow numpy python-snap7   # 真实 PLC 时需 python-snap7
"""

import argparse
import time
import sys
import os
import numpy as np

# ─── 路径设置（确保能找到 plc_simulator）────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plc_simulator import (
    S7Simulator, S7Client, SignalBuffer,
    FAULT_LABELS_CN, FRAME_SIZE, FS,
)

# ────────────────────────────────────────────────────────────────
# 信号预处理（与训练时保持一致）
# ────────────────────────────────────────────────────────────────

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score 归一化，与训练数据集预处理完全一致。

    参数
    ----
    signal : np.ndarray, shape=(2048,)

    返回
    ----
    np.ndarray, shape=(1, 2048, 1)  — CNN 输入格式
    """
    mean = signal.mean()
    std = signal.std()
    normalized = (signal - mean) / (std + 1e-8)
    return normalized.reshape(1, FRAME_SIZE, 1).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# 模型加载
# ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """加载 Keras 模型，返回 model 对象。"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"[Monitor] 模型已加载: {model_path}")
        return model
    except ImportError:
        print("[Monitor] 警告: 未安装 TensorFlow，将使用随机模拟推理（仅用于演示）")
        return None
    except Exception as e:
        print(f"[Monitor] 模型加载失败: {e}")
        return None


def predict(model, x: np.ndarray):
    """
    运行推理。若 model 为 None（无 TensorFlow），返回随机结果用于演示。

    返回
    ----
    fault_code : int     预测故障代码 (0~3)
    confidence : float   对应类别置信度 (0~1)
    probs : np.ndarray   各类别概率 shape=(4,)
    """
    if model is None:
        # 演示模式：返回随机概率
        probs = np.random.dirichlet([5, 1, 1, 1]).astype(np.float32)
    else:
        probs = model.predict(x, verbose=0)[0]

    fault_code = int(np.argmax(probs))
    confidence = float(probs[fault_code])
    return fault_code, confidence, probs


# ────────────────────────────────────────────────────────────────
# 主监控循环
# ────────────────────────────────────────────────────────────────

HEADER = (
    "\n{'─'*70}\n"
    "  西门子 S7 PLC 轴承故障实时诊断系统\n"
    f"{'─'*70}"
)

RESULT_TEMPLATE = (
    "  [{ts}]  诊断={label:<10s}  置信度={conf:.1%}  "
    "真实={truth:<10s}  {'✓' if ok else '✗'}"
)

LABEL_NAMES = [FAULT_LABELS_CN[i] for i in range(4)]


def run_monitor(
    plc,                  # S7Simulator 或 S7Client
    model,
    cycles: int = 20,
    chunk_size: int = 64,
    inter_frame_delay: float = 0.1,
    write_back: bool = False,
):
    """
    主监控循环。

    参数
    ----
    plc            : 数据源（仿真器或真实客户端）
    model          : Keras 模型（None 时使用随机演示）
    cycles         : 诊断次数
    chunk_size     : 每次从 PLC 读取的样本数（模拟批次粒度）
    inter_frame_delay : 每帧诊断间隔秒数（模拟实时采样间隔）
    write_back     : 是否将结果写回 PLC（仅 S7Client 支持）
    """
    use_simulator = isinstance(plc, S7Simulator)

    print(f"\n{'='*70}")
    print(f"  西门子 S7 PLC 轴承故障实时诊断系统")
    print(f"{'='*70}")
    print(f"  模式    : {'仿真' if use_simulator else '真实PLC'}")
    print(f"  采样率  : {FS} Hz")
    print(f"  帧长度  : {FRAME_SIZE} 点 ({FRAME_SIZE/FS*1000:.1f} ms)")
    print(f"  上传粒度: {chunk_size} 点/批次")
    print(f"  诊断次数: {cycles}")
    print(f"{'='*70}\n")

    buf = SignalBuffer(capacity=FRAME_SIZE, overlap=0)
    results = []

    col_w = [5, 12, 8, 10, 10, 6]
    header_row = (
        f"  {'帧#':>{col_w[0]}}  "
        f"{'预测结果':<{col_w[1]}}  "
        f"{'置信度':>{col_w[2]}}  "
        f"{'正常':>{col_w[3]}}  "
        f"{'内圈故障':>{col_w[3]}}  "
        f"{'外圈故障':>{col_w[3]}}  "
        f"{'滚动体故障':>{col_w[3]}}  "
        f"{'真实标签':<{col_w[4]}}  {'正确':>{col_w[5]}}"
    )
    sep = "  " + "─" * (len(header_row) - 2)
    print(header_row)
    print(sep)

    correct = 0
    for cycle_idx in range(cycles):
        buf.reset()

        # ── 数据采集（模拟 PLC 分批上传）─────────────────────────
        while True:
            if use_simulator:
                chunk = plc.read_vibration_chunk(samples=chunk_size)
            else:
                # 真实 PLC：等待就绪标志
                for _ in range(50):
                    if plc.is_ready():
                        break
                    time.sleep(0.02)
                chunk = plc.read_vibration_frame()   # 真实 PLC 一次读整帧
                break

            if buf.push(chunk):
                break

        frame = buf.get_frame()

        # ── 预处理 + 推理 ─────────────────────────────────────────
        x = preprocess_signal(frame)
        fault_code, confidence, probs = predict(model, x)

        # ── 获取真实标签（仿真专用）──────────────────────────────
        if use_simulator:
            true_code, _, true_name_cn = plc.get_true_label()
        else:
            true_code = -1
            true_name_cn = "未知"

        is_correct = (fault_code == true_code) if use_simulator else None

        # ── 打印结果行 ────────────────────────────────────────────
        pred_name = FAULT_LABELS_CN[fault_code]
        ok_mark = "✓" if is_correct else ("✗" if is_correct is False else "-")
        if is_correct:
            correct += 1

        row = (
            f"  {cycle_idx+1:>{col_w[0]}}  "
            f"{pred_name:<{col_w[1]}}  "
            f"{confidence:>{col_w[2]}.1%}  "
            + "  ".join(f"{p:>{col_w[3]}.3f}" for p in probs)
            + f"  {true_name_cn:<{col_w[4]}}  {ok_mark:>{col_w[5]}}"
        )
        print(row)
        results.append({
            "frame": cycle_idx + 1,
            "pred_code": fault_code,
            "pred_name": pred_name,
            "confidence": confidence,
            "true_code": true_code,
            "true_name": true_name_cn,
            "correct": is_correct,
        })

        # ── 结果写回 PLC ──────────────────────────────────────────
        if write_back and not use_simulator:
            try:
                plc.write_diagnosis_result(fault_code, confidence)
            except Exception as e:
                print(f"  [警告] 写回 PLC 失败: {e}")

        time.sleep(inter_frame_delay)

    # ── 汇总统计 ─────────────────────────────────────────────────
    print(sep)
    if use_simulator:
        accuracy = correct / cycles * 100
        print(f"\n  诊断完成 | 总帧数: {cycles} | 正确: {correct} | 准确率: {accuracy:.1f}%\n")
    else:
        print(f"\n  诊断完成 | 总帧数: {cycles}\n")

    return results


# ────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="西门子 S7 PLC 轴承故障实时诊断监控器"
    )
    parser.add_argument(
        "--plc-ip", default=None,
        help="PLC IP 地址（不填则使用仿真模式）"
    )
    parser.add_argument("--rack", type=int, default=0, help="PLC 机架号（默认 0）")
    parser.add_argument("--slot", type=int, default=1, help="PLC 插槽号（默认 1）")
    parser.add_argument("--db", type=int, default=100, help="数据块编号（默认 DB100）")
    parser.add_argument(
        "--fault",
        choices=["normal", "inner", "outer", "ball"],
        default="inner",
        help="仿真模式下的故障类型（默认 inner）"
    )
    parser.add_argument("--noise", type=float, default=0.05, help="仿真噪声水平（默认 0.05）")
    parser.add_argument("--cycles", type=int, default=20, help="诊断帧数（默认 20）")
    parser.add_argument("--chunk", type=int, default=64, help="PLC 每次上传样本数（默认 64）")
    parser.add_argument("--delay", type=float, default=0.05, help="帧间延迟秒数（默认 0.05）")
    parser.add_argument(
        "--model",
        default="models/bearing_cnn_model.keras",
        help="Keras 模型路径"
    )
    parser.add_argument(
        "--write-back", action="store_true",
        help="将诊断结果写回 PLC（仅真实 PLC 模式）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 切换到脚本目录，确保相对路径正确
    os.chdir(SCRIPT_DIR)

    # 加载模型
    model_path = args.model
    model = load_model(model_path)

    # 创建 PLC 数据源
    if args.plc_ip:
        plc = S7Client(
            ip=args.plc_ip,
            rack=args.rack,
            slot=args.slot,
            db_number=args.db,
        )
        if not plc.connect():
            print("无法连接到 PLC，退出。")
            sys.exit(1)
    else:
        plc = S7Simulator(
            fault_type=args.fault,
            noise_level=args.noise,
            seed=None,
        )

    # 运行监控
    try:
        results = run_monitor(
            plc=plc,
            model=model,
            cycles=args.cycles,
            chunk_size=args.chunk,
            inter_frame_delay=args.delay,
            write_back=args.write_back,
        )
    finally:
        if isinstance(plc, S7Client):
            plc.disconnect()

    return results


if __name__ == "__main__":
    main()
