# 输入与视频标注

输入`inputs_v2/{left,right}.npz`为上轮采用的观测恢复结果，本轮原样复制到`selected_inputs`。坐标米，时间秒；原生300Hz来源仍指向capture内C3D，634时刻为沿用开发采样，不是全原生300Hz重新拟合。

14列顺序：IWR、OWR、IHAND、OHAND、THM3、THM6、IDX3、IDX6、MID0、MID6、RNG3、RNG6、PNK3、PNK6。`markers_world_m`是真实观测或有来源的短插值；`anatomical_to_raw_columns`与`anatomical_source_labels`保留关联来源；`marker_provenance`为0缺失、1原生、2插值。上一轮重关联身份仍是算法推断，不能称独立真值。隔离三采样不恢复伪观测。

输出`selected_review/{left,right}_fit.npz`保存pose、beta、wrist、orient、vertices_world_m、joints_world_m、markers_predicted_world_m等。左手新增`stability_adopted_frames`表示纯三维保护性回退后采用候选的522/634时刻；其他112时刻pose精确沿用基线。右手整文件沿用基线。共享三角面/重心坐标绑定及高度不变；marker仍为10mm背侧半球，偏心点必须贴在mesh面内。0mm为沿用的有效光学中心假设，不是实测半球高度。

视频严格两栏：左原RGB加C3D投影，右同一RGB加最新左MANO。黄色空心圈=原生C3D，橙色菱形=短插值，紫色叉=拟合中心，蓝点=当前mesh贴面点；星号=重关联身份，@=来源为原生未命名轨迹。不是RGB检测点，也不意味着已证实外参/同步绝对正确。沿用最新米制sxt→RGB外参，不使用旧外参。

每段同名`marker_tracks_*.csv`含RGB索引、C3D时间、手别、解剖列、原始来源列/标签、provenance、世界坐标米、观测/预测像素、残差毫米。缺失点留缺失，不以预测冒充测量。相机裁剪只改变两栏共同显示范围，不改变pose。0.5倍播放，约3.83–4秒原动作，无合成动作帧，时间戳按真实RGB间隔编码。
