"""
plc_simulator.py
================
西门子 S7 PLC 振动数据采集模拟器

本模块模拟一台西门子 S7-1200/S7-1500 PLC 通过加速度传感器
采集轴承振动信号的完整过程，包含：

  - S7Client：封装 python-snap7 接口，支持真实 PLC 连接
  - S7Simulator：在无真实 PLC 时生成仿真振动信号（4 种故障类型）
  - SignalBuffer：循环缓冲区，积累够 2048 点后触发一次诊断

典型 DB 块布局（PLC 侧）:
  DB100.DBD0  ~ DB100.DBD8188  → 2048 个 REAL（4 字节 / 点）= 8192 字节
  DB100.DBW8192                 → WORD：当前故障代码（PLC 自身逻辑写入，可选）
  DB100.DBX8194.0               → BOOL：新数据就绪标志

使用示例
--------
  from plc_simulator import S7Simulator, SignalBuffer

  plc = S7Simulator(fault_type='inner', noise_level=0.05)
  buf = SignalBuffer(capacity=2048)

  while True:
      chunk = plc.read_vibration_chunk(samples=64)   # 模拟每次 PLC 上传 64 点
      if buf.push(chunk):                             # 缓冲满 2048 点
          signal = buf.get_frame()                    # 取出完整帧
          # → 送入 CNN 模型诊断
          break
"""

import time
import struct
import numpy as np
from typing import Optional, Tuple

# ────────────────────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────────────────────
FAULT_TYPES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
    3: "Ball Fault",
}

FAULT_LABELS_CN = {
    0: "正常",
    1: "内圈故障",
    2: "外圈故障",
    3: "滚动体故障",
}

# 轴承特征频率（相对于转频的倍率，基于 CWRU 6205-2RS 轴承参数）
BEARING_FREQ_RATIO = {
    "inner": 5.415,   # 内圈故障特征频率
    "outer": 3.585,   # 外圈故障特征频率
    "ball":  2.357,   # 滚动体故障特征频率
}

FS = 12_000          # 采样频率 12 kHz，与 CWRU 数据集一致
SHAFT_RPM = 1797     # 电机转速（rpm）
FRAME_SIZE = 2048    # 每次诊断使用的采样点数


# ────────────────────────────────────────────────────────────────
# S7 真实连接客户端（需安装 python-snap7）
# ────────────────────────────────────────────────────────────────
class S7Client:
    """
    封装 python-snap7，连接真实西门子 S7 PLC。

    参数
    ----
    ip : str
        PLC 的 IP 地址，例如 '192.168.0.1'
    rack : int
        机架号，S7-1200/1500 一般为 0
    slot : int
        插槽号，S7-1200 为 1，S7-1500 为 0
    db_number : int
        存放振动数据的 DB 块编号，默认 DB100
    """

    def __init__(self, ip: str, rack: int = 0, slot: int = 1, db_number: int = 100):
        try:
            import snap7
            self._snap7 = snap7
        except ImportError:
            raise ImportError(
                "请先安装 python-snap7：pip install python-snap7\n"
                "同时需要在系统中安装 snap7 动态库（libsnap7.so / snap7.dll）"
            )
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.db_number = db_number
        self._client = snap7.client.Client()

    def connect(self) -> bool:
        """建立 TCP 连接到 PLC，返回是否成功。"""
        try:
            self._client.connect(self.ip, self.rack, self.slot)
            print(f"[S7Client] 已连接到 PLC @ {self.ip}")
            return self._client.get_connected()
        except Exception as e:
            print(f"[S7Client] 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接。"""
        self._client.disconnect()
        print("[S7Client] 已断开连接")

    def is_ready(self) -> bool:
        """读取 DB100.DBX8194.0，检查新数据就绪标志。"""
        try:
            data = self._client.db_read(self.db_number, 8194, 1)
            return bool(data[0] & 0x80)   # Bit 0 of byte 8194
        except Exception:
            return False

    def read_vibration_frame(self) -> np.ndarray:
        """
        一次性读取完整的 2048 点振动帧（8192 字节 REAL 数组）。
        返回 shape=(2048,) 的 float32 数组。
        """
        raw = self._client.db_read(self.db_number, 0, FRAME_SIZE * 4)
        floats = [
            struct.unpack(">f", raw[i * 4: i * 4 + 4])[0]
            for i in range(FRAME_SIZE)
        ]
        # 清除就绪标志（写 0 到 DBX8194.0）
        self._client.db_write(self.db_number, 8194, bytearray([0x00]))
        return np.array(floats, dtype=np.float32)

    def write_diagnosis_result(self, fault_code: int, confidence: float):
        """
        将诊断结果写回 PLC：
          DB100.DBW8196  → WORD：故障代码（0=正常，1/2/3=各类故障）
          DB100.DBD8198  → REAL：置信度（0.0~1.0）
        """
        code_bytes = struct.pack(">H", fault_code)
        conf_bytes = struct.pack(">f", confidence)
        self._client.db_write(self.db_number, 8196, bytearray(code_bytes))
        self._client.db_write(self.db_number, 8198, bytearray(conf_bytes))


# ────────────────────────────────────────────────────────────────
# S7 仿真器（无需真实 PLC）
# ────────────────────────────────────────────────────────────────
class S7Simulator:
    """
    模拟西门子 S7 PLC 的振动信号采集，无需真实硬件。

    生成的信号由以下分量叠加：
      - 转频基波及谐波
      - 故障特征频率冲击（调幅模型）
      - 可调高斯白噪声

    参数
    ----
    fault_type : str
        故障类型，可选 'normal' / 'inner' / 'outer' / 'ball'
    noise_level : float
        噪声强度（相对于信号幅值的比例），默认 0.05
    shaft_rpm : float
        轴转速（rpm），默认 1797
    fs : int
        采样频率（Hz），默认 12000
    """

    FAULT_TYPE_MAP = {
        "normal": 0,
        "inner":  1,
        "outer":  2,
        "ball":   3,
    }

    def __init__(
        self,
        fault_type: str = "normal",
        noise_level: float = 0.05,
        shaft_rpm: float = SHAFT_RPM,
        fs: int = FS,
        seed: Optional[int] = None,
    ):
        if fault_type not in self.FAULT_TYPE_MAP:
            raise ValueError(f"fault_type 必须是 {list(self.FAULT_TYPE_MAP.keys())} 之一")
        self.fault_type = fault_type
        self.fault_code = self.FAULT_TYPE_MAP[fault_type]
        self.noise_level = noise_level
        self.shaft_rpm = shaft_rpm
        self.fs = fs
        self._rng = np.random.default_rng(seed)
        self._sample_cursor = 0   # 连续时间轴游标，保证跨帧相位连续

        self._f_shaft = shaft_rpm / 60.0   # 转频（Hz）
        print(
            f"[S7Simulator] 初始化完成 | 故障类型: {fault_type} "
            f"({FAULT_LABELS_CN[self.fault_code]}) | "
            f"采样率: {fs} Hz | 噪声: {noise_level}"
        )

    # ── 内部信号生成 ─────────────────────────────────────────────

    def _generate_samples(self, n: int) -> np.ndarray:
        """生成 n 个连续振动样本点。"""
        t = (np.arange(n) + self._sample_cursor) / self.fs
        signal = np.zeros(n)

        # 1. 转频基波 + 三次谐波（机械不平衡）
        for k in [1, 2, 3]:
            amp = 0.5 / k
            signal += amp * np.sin(2 * np.pi * k * self._f_shaft * t)

        # 2. 故障特征冲击
        if self.fault_type != "normal":
            ratio = BEARING_FREQ_RATIO[self.fault_type]
            f_fault = ratio * self._f_shaft

            # 调幅模型：冲击序列 × 衰减指数包络
            impulse = np.sin(2 * np.pi * f_fault * t)
            # 模拟每次冲击后的指数衰减（近似）
            decay_period = int(self.fs / f_fault)
            envelope = np.zeros(n)
            for start in range(0, n, max(1, decay_period)):
                end = min(start + decay_period, n)
                length = end - start
                envelope[start:end] = np.exp(-10 * np.arange(length) / max(1, decay_period))

            signal += 1.5 * impulse * envelope

        # 3. 高斯白噪声
        noise_std = self.noise_level * np.std(signal + 1e-8)
        signal += self._rng.normal(0, noise_std, n)

        self._sample_cursor += n
        return signal.astype(np.float32)

    # ── 公开接口 ─────────────────────────────────────────────────

    def read_vibration_chunk(self, samples: int = 64, delay: float = 0.0) -> np.ndarray:
        """
        模拟 PLC 每次上传一批振动样本（模拟 DMA/中断驱动上传）。

        参数
        ----
        samples : int
            本次返回的样本数，模拟 PLC 上传粒度
        delay : float
            模拟网络/总线延迟（秒），0 表示无延迟

        返回
        ----
        np.ndarray, shape=(samples,), dtype=float32
        """
        if delay > 0:
            time.sleep(delay)
        return self._generate_samples(samples)

    def read_full_frame(self) -> np.ndarray:
        """直接返回一个完整的 2048 点帧（用于快速测试）。"""
        return self._generate_samples(FRAME_SIZE)

    def get_true_label(self) -> Tuple[int, str, str]:
        """返回当前模拟的真实故障类型（仿真专用）。"""
        return self.fault_code, self.fault_type, FAULT_LABELS_CN[self.fault_code]

    def set_fault_type(self, fault_type: str):
        """运行时切换故障类型（模拟工况变化）。"""
        if fault_type not in self.FAULT_TYPE_MAP:
            raise ValueError(f"无效的故障类型: {fault_type}")
        self.fault_type = fault_type
        self.fault_code = self.FAULT_TYPE_MAP[fault_type]
        print(f"[S7Simulator] 故障类型已切换为: {fault_type} ({FAULT_LABELS_CN[self.fault_code]})")


# ────────────────────────────────────────────────────────────────
# 信号缓冲区
# ────────────────────────────────────────────────────────────────
class SignalBuffer:
    """
    循环缓冲区：积累来自 PLC 的小批量数据，满 `capacity` 点后可提取诊断帧。

    参数
    ----
    capacity : int
        帧长度，默认 2048（与模型输入对齐）
    overlap : int
        相邻帧之间的重叠点数，0 表示不重叠（仅保留滑窗时有用）
    """

    def __init__(self, capacity: int = FRAME_SIZE, overlap: int = 0):
        self.capacity = capacity
        self.overlap = overlap
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._filled = 0

    def push(self, chunk: np.ndarray) -> bool:
        """
        向缓冲区追加数据。

        返回 True 表示缓冲区已满，可以调用 get_frame()。
        如果 chunk 超过剩余空间，多余部分会被截断（下一帧使用）。
        """
        n = len(chunk)
        space = self.capacity - self._filled
        take = min(n, space)
        self._buffer[self._filled: self._filled + take] = chunk[:take]
        self._filled += take

        if self._filled >= self.capacity:
            return True
        return False

    def get_frame(self) -> np.ndarray:
        """
        取出完整帧并重置缓冲区（保留 overlap 个点用于下一帧）。
        返回 shape=(capacity,) 的 float32 数组。
        """
        frame = self._buffer.copy()
        if self.overlap > 0:
            self._buffer[: self.overlap] = self._buffer[self.capacity - self.overlap:]
            self._filled = self.overlap
        else:
            self._buffer[:] = 0
            self._filled = 0
        return frame

    def reset(self):
        """清空缓冲区。"""
        self._buffer[:] = 0
        self._filled = 0

    @property
    def fill_ratio(self) -> float:
        """当前填充比例（0.0 ~ 1.0）。"""
        return self._filled / self.capacity


# ────────────────────────────────────────────────────────────────
# 快速自测
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("S7 PLC 模拟器自测")
    print("=" * 60)

    for fault in ["normal", "inner", "outer", "ball"]:
        sim = S7Simulator(fault_type=fault, noise_level=0.05, seed=42)
        buf = SignalBuffer(capacity=FRAME_SIZE)

        chunk_size = 64
        steps = 0
        while True:
            chunk = sim.read_vibration_chunk(samples=chunk_size)
            steps += 1
            if buf.push(chunk):
                break

        frame = buf.get_frame()
        code, name_en, name_cn = sim.get_true_label()
        print(
            f"  [{fault:6s}] 帧形状={frame.shape}  "
            f"均值={frame.mean():.4f}  标准差={frame.std():.4f}  "
            f"上传批次={steps}  标签={code}({name_cn})"
        )

    print("\n自测完成 ✓")
