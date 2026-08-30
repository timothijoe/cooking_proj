# Retargeting 总章程

Retargeting 把原始手套骨架转换为规范、可验证的 Wuji 20 关节轨迹。原始数据不可原地修改；
每次转换必须保留时间语义、坐标/左右手约定、关节顺序、限位与有限值检查，并把输出写入
被 Git 忽略的数据目录。

离线转换不应连接设备。实体回放属于真机能力，必须经过独立预检和授权。当前实现详见
[手套到 Wuji](evolution/glove-to-wuji.md) 与 [贴桌后退](evolution/table-retreat.md)。
