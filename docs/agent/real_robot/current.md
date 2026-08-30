# 真机当前状态

Tianji 保留诊断、离线 sampled-IK、位置/阻抗切菜、轨迹平移、键盘/手柄点动和双臂批流；
Wuji 保留只读 preflight、姿态/复位/半握拳/轨迹发送工具；联合层保留双臂与手的 batch
stream/playback。

这些文件已迁移且自动测试通过，但迁移后没有连接 Tianji 或 Wuji 实体设备。因此当前只可
宣称“接口和离线路径已保留”，不能宣称真机运动已验收。控制模式、轨迹字段和设备接入
分别见对应纪传体；旧文档吸收情况见[来源映射](source-map.md)。人工入口和安全分级见
[完整真机教程](../../../tutorials/hardware/README.md)。
