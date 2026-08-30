# Wuji Hand 工具

`tianji-robot hardware wuji-sdk preflight` 只读取并验证轨迹，不搜索、连接、使能或控制实体手。
底层 `SdkWujiHand` 采用显式 `connect → arm → command → disarm → close` 生命周期，构造时
不得隐式连接。

`wuji_hand.tools` 还保留发送姿态、轨迹、复位、半握拳和单关节测试。它们属于高风险实验
入口，使用前必须核对设备序列号、左右手、关节顺序、限位、温度/错误码、缓入和异常去使能。
迁移后尚无实体设备验证。
