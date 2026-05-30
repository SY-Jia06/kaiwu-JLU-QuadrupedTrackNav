# 项目简介
## 任务目标
四足机器人自主导航运控赛题。选手需要使用强化学习算法训练智能体，让其控制GO2机器狗在仿真环境内探索并学习自主导航与运动控制策略，使得机器狗可以在 未知/半未知 场景中实现 自主寻路，以尽可能短的时间跨越地形，同时保持运动的稳定性。

任务包含两种模式：

标准模式（Standard）：机器人分别在多种复杂地形（坡面、楼梯、迷宫）上行走，以前进距离、通过时间、能量效率和姿态稳定性综合评分
赛道模式（Track）：机器人在由多种子地形串联构成的赛道上从起点导航至终点，以完成数量、通过时间、姿态稳定性和能量效率综合评分
选手需要设计和优化强化学习算法，使四足机器人能够：

在各类地形上保持稳定行走（运动控制）
自主规划路径并避开障碍物（自主导航）
高效、快速地从起点到达终点（赛道模式）
## 环境介绍

### 地形
赛题使用 trimesh 类型地形，分为内置地形和自研地形两类。standard 模式下各子地形按比例分配；track 模式下子地形串联成单向赛道。

#### 标准模式地形
地形类型	说明	图例
pyramid_slope	金字塔坡面，向上	
pyramid_slope_inv	金字塔坡面，向下	
pyramid_stairs	金字塔楼梯，向上	
pyramid_stairs_inv	金字塔楼梯，向下	
maze	迷宫地形，随机生成高度为 0.5m 的障碍物，地形中至少保持一条从进入边到对边的通路	
#### 赛道模式地形
地形类型	说明	图例
track	由多种子地形串联构成的单向赛道：前段为 pyramid_slope、pyramid_slope_inv、pyramid_stairs、pyramid_stairs_inv 的任意排列，末段必须为 open_entry_maze（赛道终点恒为迷宫）	
open_entry_maze	赛道专用终点地形：迷宫出入口均为开放通道，机器人从进入边行至对边的出口即完成赛道	
### 元素介绍
元素	说明
Go2 四足机器人	Unitree Go2 四足机器人，具有 4 条腿 × 3 关节（hip/thigh/calf）共 12 个可控关节
地形	多种复杂地形（坡面、楼梯、迷宫、赛道），用于测试机器人的运动能力和导航能力
在创建训练任务和评估任务时，上述地形的配置方式有所不同。具体请查看开发指南-环境配置部分。

## 计分规则

### Standard 模式
以最稳定、最节能的方式尽可能走远（以走穿地形为目标）。

总分
=
0.4
×
Score
forward
+
0.2
×
Score
time
+
0.2
×
Score
energy
+
0.2
×
Score
posture
总分=0.4×Score 
forward
​
 +0.2×Score 
time
​
 +0.2×Score 
energy
​
 +0.2×Score 
posture
​
 
子项	权重	含义
前进距离分数	0.4	从地形块中心出生，按"与出生点的 2D 欧氏距离 / 半块长度"归一化；走到半块即满分（= 走穿：从中心开始穿过半个地形长度）
时间分数	0.2	仅走穿地形的模型获得，用时越短分越高
能耗分数	0.2	episode 平均单步关节机械功率的指数衰减，越节能越高
姿态分数	0.2	episode 平均 roll/pitch 偏移的指数衰减，越平稳越高
### Track 模式
在限定步数内，以最快、最稳定、最节能的方式从起点到达目标点。

总分
=
完成系数
×
(
 
0.4
×
Score
time
+
0.4
×
Score
posture
+
0.2
×
Score
energy
 
)
总分=完成系数×(0.4×Score 
time
​
 +0.4×Score 
posture
​
 +0.2×Score 
energy
​
 )
其中完成系数为当前任务中 完成赛道的机器狗数量占总数的比例：单只机器狗完成赛道记为 1、失败或超时记为 0，对批次内所有机器狗求均值即可。例如并行 1024 只机器狗、其中 512 只走到终点，完成系数 = 0.5。

子项	权重	含义
时间分数	0.4	用时越短分越高
姿态分数	0.4	episode 平均 roll/pitch 偏移的指数衰减
能耗分数	0.2	episode 平均单步关节机械功率的指数衰减
### 注意事项
任务失败条件：主体或关节接触地面（姿态异常 / 摔倒）
任务超时条件：达到最大步数（episode_length_s 对应的步数）仍未完成
Standard 模式走穿判定：从地形块中心出生，
∥
pos
current
−
pos
spawn
∥
2
≥
L
terrain
/
2
−
0.1
∥pos 
current
​
 −pos 
spawn
​
 ∥ 
2
​
 ≥L 
terrain
​
 /2−0.1（2D 欧氏距离，方向无关）；默认 
L
terrain
=
8
 
m
L 
terrain
​
 =8m，阈值约 3.9 m
Track 模式完成判定：机器人从起点到达终点（目标点）
Sim 赛段：使用 height scan 观测，无需深度视觉相机

# 环境详述
## 环境配置
在智能体和环境的交互中，首先会调用 env.reset 方法，该方法接受一个 usr_conf 参数，通过读取 train_env_conf_standard_locomotion.toml 文件的内容来实现定制化的环境配置。用户可以通过修改该 TOML 文件中的内容来调整环境配置。

```python
# usr_conf 为用户传入的环境配置
reset_data = env.reset(usr_conf)
obs, critic_obs = reset_data
```

train_env_conf_standard_locomotion.toml 为示例环境配置，用于在标准模式训练四足机器人的运动控制能力。

环境配置中包含以下配置信息：

配置项	类型	合法范围	说明
env.num_envs	int	[1, 4096]	并行环境数量
env.episode_length_s	float	>0	最大 episode 时长（秒）
terrain.mode	string	"standard" | "track"	地形模式
terrain.num_rows	int	[1, 10]	难度级别数（沿 X 轴的课程档位）
terrain.num_cols	int	[1, 40]	同一难度下的并行地块数（沿 Y 轴）
terrain.difficulty_range	list[float]	[0.0, 1.0]	难度范围
terrain.curriculum	bool	true/false	是否启用地形课程学习
terrain.max_init_terrain_level	int	[0, 9]	机器人初始放置的最大难度档
terrain.standard..proportion	float	[0, 1]	各子地形比例（总和须为 1.0）
domain_rand.enable_domain_rand	bool	true/false	域随机化总开关
domain_rand.randomize_friction	bool	true/false	是否随机化地面摩擦系数
domain_rand.friction_range	list[float]	≥0	摩擦系数采样范围
domain_rand.push_robots	bool	true/false	是否周期性施加外部推力
domain_rand.push_interval_s	float	>0	推力间隔（秒）
domain_rand.max_push_vel_xy	float	≥0	XY 平面最大推力速度 (m/s)
noise.add_noise	bool	true/false	是否在观测中加入噪声
init_state.pos	list[float]	z: [0.30, 0.60]	机器人初始位置 [x, y, z] (m)
commands.resampling_time	list[float]	>0	速度命令重采样区间 [min, max]（秒）
commands.limit.lin_vel_x	list[float]	—	X 方向线速度采样上限
commands.limit.lin_vel_y	list[float]	—	Y 方向线速度采样上限
commands.limit.ang_vel_z	list[float]	—	偏航角速度采样上限
commands.ranges.lin_vel_x	list[float]	—	X 方向线速度初始采样范围
commands.ranges.lin_vel_y	list[float]	—	Y 方向线速度初始采样范围
commands.ranges.ang_vel_yaw	list[float]	—	偏航角速度初始采样范围
rewards..weight	float	—	各奖励项权重
rewards..params.*	—	—	各奖励项参数
💡 补充说明：

train_env_conf_standard_locomotion.toml 文件中的配置仅在训练时生效，请按上表数据描述进行配置。若配置错误，训练任务会变为"失败"状态，此时可以通过查看 env 模块的错误日志进行排查。
若需调整模型评估任务时的配置，用户需要通过腾讯开悟平台创建评估任务并完成环境配置，详细参数见智能体模型评估模式。
train_env_conf_standard_locomotion.toml 采用的默认配置：

```toml
[env]
num_envs = 2048
episode_length_s = 25

[env_conf]
seed = 0

[terrain]
mode = "standard"
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 5

[terrain.standard.pyramid_slope]
proportion = 0.15

[terrain.standard.pyramid_slope_inv]
proportion = 0.2

[terrain.standard.pyramid_stairs]
proportion = 0.25

[terrain.standard.pyramid_stairs_inv]
proportion = 0.3

[terrain.standard.maze]
proportion = 0.1

[domain_rand]
enable_domain_rand = true
randomize_friction = true
friction_range = [0.3, 1.5]
push_robots = true
push_interval_s = 15
max_push_vel_xy = 0.5

[noise]
add_noise = true

[init_state]
pos = [0.0, 0.0, 0.35]

[commands]
resampling_time = [10.0, 10.0]

[commands.limit]
lin_vel_x = [-2.0, 2.0]
lin_vel_y = [-1.5, 1.5]
ang_vel_z = [-1.5, 1.5]

[commands.ranges]
lin_vel_x = [0.0, 0.5]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-1.0, 1.0]

[rewards.track_lin_vel_xy]
weight = 1.0
[rewards.track_lin_vel_xy.params]
std = 0.25
command_name = "base_velocity"

[rewards.track_ang_vel_z]
weight = 0.5
[rewards.track_ang_vel_z.params]
std = 0.25
command_name = "base_velocity"

[rewards.lin_vel_z]
weight = -2.0

[rewards.ang_vel_xy]
weight = -0.05

[rewards.joint_acc]
weight = -2.5e-7

[rewards.joint_torques]
weight = -1e-4

[rewards.dof_pos_limits]
weight = -2.0

[rewards.action_rate]
weight = -0.01

[rewards.undesired_contacts]
weight = -1.0
[rewards.undesired_contacts.params]
threshold = 1

[rewards.flat_orientation]
weight = -1.5

[rewards.reach_goal]
weight = 10.0
[rewards.reach_goal.params]
threshold = 0.6
```

Standard模式下的合法子地形类型：pyramid_slope | pyramid_slope_inv | pyramid_stairs | pyramid_stairs_inv | maze

### 切换到 Track 模式
进入导航阶段训练时，需把 terrain 段替换为 track 配置，并参考 LocomotionConfig 自行设计新的训练阶段配置。

```toml
[terrain.track]
track_length = 5
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
```


Track模式下的合法子地形类型：pyramid_slope | pyramid_slope_inv | pyramid_stairs | pyramid_stairs_inv | open_entry_maze

需要注意open_entry_maze必须配在赛道最后，否则训练会报错


## 环境信息
数据名	数据类型	数据描述
frame_no	int	当前交互帧号
obs	torch.Tensor	策略观测 (num_envs, obs_dim)
rewards	torch.Tensor	当前步总 reward (num_envs,)
terminated	torch.Tensor[bool]	真实终止（摔倒、目标达成）
truncated	torch.Tensor[bool]	超时截断
infos	dict	Isaac Lab / RSL-RL extras
privileged_obs	torch.Tensor | None	critic 观测 (num_envs, critic_obs_dim)
### 奖励信息（reward）
reward 是 Isaac Lab 按 TOML 配置的 [rewards.*] 段实时计算出的每一步奖励总和，shape = (num_envs,)。

注意：reward 是强化学习训练的驱动信号，由代码包默认激活的 11 项 reward（见"环境监控信息-Reward 指标"一节）加权求和得到。它不等于平台评分系统的"总分"——总分由默认的监控信息进行上报。

### 观测信息（observation）
策略观测 obs 是传递给 Actor 网络的输入，布局如下：

obs = [proprio(45) | height_scan(256)] → 301 维

proprio（45 维）字段布局：

区间	维度	含义	来源
[0:3]	3	base_ang_vel，机体角速度，scale=0.25	Isaac Lab mdp
[3:6]	3	projected_gravity，重力方向投影	Isaac Lab mdp
[6:9]	3	velocity_commands (vx, vy, wz)	command manager
[9:21]	12	joint_pos_rel，关节相对默认位置	robot data
[21:33]	12	joint_vel_rel，关节速度，scale=0.05	robot data
[33:45]	12	last_action，上一帧动作	action manager
height_scan（256 维）：

字段名	区间	类型	说明
height_scan	obs[:, 45:301]	torch.Tensor	16×16 前方高度扫描，clip [-5, 5]，scale=2.5
privileged_obs（316 维）是传递给 Critic 网络的观测，在 proprio 基础上额外包含 base_lin_vel（机体线速度）和 joint_effort（关节力矩）等特权信息，仅训练时使用，体现"不对称 Actor-Critic"设计。

💡 补充说明：track 地形下环境额外提供 env.goal_positions（目标点世界坐标）和 env.goal_yaw（目标点朝向），以及 env.scene.sensors["nav_scanner"]（前瞻遮挡扫描），选手可从这些属性构造导航特征并拼接到 obs。目标点观测详细信息请参考：Go2 SDK 开发指南

额外信息（infos）
infos 是一个 dict，包含仿真环境给的辅助信息。

### 动作空间
Go2 为 12 自由度四足机器人，动作空间为 12 维连续动作：

字段名	类型	Shape	取值范围	说明
actions	float32	(num_envs, 12)	[-1.0, 1.0]	12 个关节控制动作
#### 合法动作
动作值为归一化的偏移量，经过 action_scale（默认 0.25）缩放后加到默认关节角度上，作为 PD 控制器的目标角度。关节维度对应：

维度	关节组	说明
0~2	前左腿	hip / thigh / calf
3~5	前右腿	hip / thigh / calf
6~8	后左腿	hip / thigh / calf
9~11	后右腿	hip / thigh / calf
注意：具体关节顺序以 Isaac Lab Unitree Go2 资产配置为准。

### 时间信息
步（step）和帧（frame）存在一一对应关系。

每一步中，智能体选择一个 12 维动作，环境据此更新状态并返回新的观测、奖励和终止信号。env.step() 返回的 frame_no 即当前交互帧号。


## 环境监控信息
监控面板中包含 env 模块，表示环境指标数据，每 1 分钟采集最新结束的 episode 数据、求平均后展示。详细说明如下：

### Standard 模式
#### 全局环境指标
面板名称	指标名称	说明
已结束任务数	completed_count	正常完成的 episode 数
已结束任务数	abnormal_count	异常终止的 episode 数
已结束任务数	timeout_count	超时终止的 episode 数
得分	total_score	单局总分均值
得分	distance_score	单局前进距离分均值
得分	time_score	单局时间分均值
得分	energy_score	单局能耗分均值
得分	pose_score	单局姿态分均值
步数	step	单局平均步数
#### [terrain_type] 指标（按地形类型分 Tab）
面板名称	指标命名规律	说明
地形-完成数	completed_count_[terrain_type]	该地形正常完成 episode 数
地形-失败数	abnormal_count_[terrain_type]	该地形异常终止 episode 数
地形-超时数	timeout_count_[terrain_type]	该地形超时终止 episode 数
地形-总分	total_score_[terrain_type]	该地形总分均值
地形-距离分数	distance_score_[terrain_type]	该地形前进距离分均值
地形-时间分数	time_score_[terrain_type]	该地形时间分均值
地形-能耗分数	energy_score_[terrain_type]	该地形能耗分均值
地形-姿态分数	pose_score_[terrain_type]	该地形姿态分均值
地形-步数	step_[terrain_type]	该地形平均步数
[terrain_type] 需替换为具体地形名称（如 pyramid_slope、maze 等），多种地形对应多个 Tab。

### Track模式
#### 全局环境指标
面板名称	指标名称	说明
已结束任务数	completed_count	正常完成的 episode 数
已结束任务数	abnormal_count	异常终止的 episode 数
已结束任务数	timeout_count	超时终止的 episode 数
得分	total_score	单局总分均值
得分	energy_score	单局能耗分均值
得分	pose_score	单局姿态分均值
得分	time_score	单局时间分均值（底层指标 key 为 kaiwu_step_score）
步数	step_avg	单局平均步数
Reward 均值	reward_mean	单局平均奖励
Reward 均值	reward_std	单局奖励标准差
#### Track 赛道-难度档
面板名称	指标命名规律	说明
赛道-完成数	completed_count_track_l{0~9}	各难度档正常完成 episode 数
赛道-失败数	abnormal_count_track_l{0~9}	各难度档异常终止 episode 数
赛道-超时数	timeout_count_track_l{0~9}	各难度档超时终止 episode 数
赛道-总分	total_score_track_l{0~9}	各难度档总分均值
赛道-能耗分数	energy_score_track_l{0~9}	各难度档能耗分均值
赛道-姿态分数	pose_score_track_l{0~9}	各难度档姿态分均值
赛道-时间分数	time_score_track_l{0~9}	各难度档时间分均值
#### Reward 指标
代码包默认激活的 reward 对应的监控面板：

面板名称	指标名称	说明
线速度跟踪奖励	reward_track_lin_vel_xy	XY 速度命令跟踪奖励均值
角速度跟踪奖励	reward_track_ang_vel_z	yaw 角速度命令跟踪奖励均值
安全奖励	reward_undesired_contacts	非脚掌接触惩罚均值
安全奖励	reward_dof_pos_limits	关节位置极限惩罚均值
平坦姿态奖励	reward_flat_orientation	非直立姿态惩罚均值
到达目标	reward_reach_goal	到达目标点奖励均值（仅 track 地形且 env.goal_positions 被维护时生效）
选手若在 reward_process.py 新增了 reward，需要在 agent_ppo/conf/monitor_builder.py 中参考 Group 2 的示例自行添加对应面板。

# 智能体详述
我们在代码包中提供了智能体的简单实现，本文将对该部分内容进行讲解，包括特征处理和奖励处理等。

## 观测处理
环境返回的 observation 信息包含了针对智能体的局部观测，可以在 PolicyObservationProcess 中进行处理。代码包默认直接透传 self.default_observation() 返回的 301 维张量，选手可在此基础上做特征工程（归一化、聚合、拼接额外特征等）：

```python
class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        # TODO (track 地形)：可按需从 env.goal_positions / env.goal_yaw
        # 或 env.scene.sensors["nav_scanner"] 构造特征并拼接到 obs。
        return obs
```

观测数据布局：

区间	维度	含义	来源
[0:3]	3	base_ang_vel，机体角速度，scale=0.25	Isaac Lab mdp
[3:6]	3	projected_gravity，重力方向投影	Isaac Lab mdp
[6:9]	3	velocity_commands，速度命令 (vx, vy, wz)	command manager
[9:21]	12	joint_pos_rel，关节相对默认位置	robot data
[21:33]	12	joint_vel_rel，关节速度，scale=0.05	robot data
[33:45]	12	last_action，上一帧动作	action manager
[45:301]	256	height_scan，16×16 前方高度扫描	height_scanner
Track 地形扩展：环境额外提供 env.goal_positions（出口位置）、env.goal_yaw（出口朝向）和 env.scene.sensors["nav_scanner"]（前瞻遮挡扫描）。

## 特征处理
当前代码包中 PolicyObservationProcess 直接使用 default_observation() 返回的原始观测，未做额外特征工程处理。

```python
def process(self):
    obs = self.default_observation()
    return obs
```

选手可以在此基础上进行特征工程扩展。

## 奖励处理
代码包提供了两个示例奖励函数，位于 reward_process.py：

奖励说明：

奖励名称	作用	类型
_reward_reach_goal	机器人与出口距离小于 0.6 m 时给予 1.0 的稀疏奖励	sparse
_reward_forward_velocity	鼓励机器人沿本体 x 轴方向前进的 dense 奖励	dense
其余通用 locomotion 奖励继承自 RewardProcessBase，在 TOML 配置中激活 [rewards.<name>] 即可，无需重复实现。选手若需训练导航策略，请在 reward_process.py 自行添加更多奖励函数。

## 算法介绍
代码包提供了基于 PPO（Proximal Policy Optimization）的 Actor-Critic 算法实现。

算法	文件路径	说明
PPO + ActorCritic	model/actor_critic.py	扁平 MLP Actor-Critic 网络，Gaussian 动作分布
### ActorCritic 网络结构
采用独立的 Actor 和 Critic 两个 MLP 网络：

组件	输入维度	隐藏层	输出维度	特殊设计
Actor	301（policy obs）	[512, 256, 128]	12（关节动作）	ELU 激活
Critic	316（critic obs）	[512, 256, 128]	1（状态价值）	LayerNorm + ELU 激活
### 训练超参数
参数	默认值	说明
lr	3e-4	学习率
num_learning_epochs	5	每次更新的 epoch 数
num_mini_batches	4	mini-batch 数量
num_steps_per_env	48	每个环境采集步数
model_save_interval	500	模型保存间隔（迭代次数）

## 算法监控信息
算法上报了 reward 等指标，用户可以通过腾讯开悟平台/客户端的监控功能查看。

针对当前算法的指标说明如下：

### 算法指标组
面板名称	指标 Key	类型	说明
总损失	total_loss	line	PPO 总 loss（policy_loss + value_loss - entropy_loss）
价值损失	value_loss	line	Critic 网络价值函数损失
策略损失	policy_loss	line	Actor 网络策略梯度损失
熵损失	entropy_loss	line	策略熵，衡量探索程度
### 奖励指标组
面板名称	指标 Key	类型	说明
线速度跟踪奖励	reward_track_lin_vel_xy	line	XY 平面速度命令跟踪奖励
说明：以上为代码包 monitor_builder.py 中定义的自定义指标。其余通用 reward 指标（如速度跟踪、姿态、步态等）由 tools/conf/monitor_default.yaml 和 tools/conf/monitor_default_track.yaml 负责展示。


## 模型保存限制策略
为了避免用户保存模型的频率过于频繁，开悟平台对模型保存会有安全限制，不同的任务会有不同的限制，限制规则详情如下：

参数	值	说明
model_save_interval	500	每 500 次训练迭代保存一次模型
注意：选手可在 conf.py 中修改 model_save_interval 的值调整保存频率，但平台侧可能有额外的最小间隔限制。


## 模型评估模式
用户可以在腾讯开悟平台上创建模型评估任务。创建任务时，可以对该任务的环境进行配置，配置信息如下：

```toml
[env]
# Number of parallel environments | 并行环境数量
num_envs = 16
# Episode length in seconds | 单回合时长(秒)
episode_length_s = 20.0
# Task type: standard / track | 任务类型
task = "standard"

[terrain]
# Terrain mode: standard / track | 地形模式
mode = "standard"
# Difficulty levels to evaluate (0=easiest, 9=hardest)
# standard 模式下: num_rows = max(level)+1 自动推导, num_cols = len(sub_terrains)
# track    模式下: num_parallel_tracks 默认 10; 仅当 max(level)+1 > 10 时才自动扩展
# 评估的难度档列表(0最简单, 9最难)
level = [0, 1, 3, 4]
# Force-disable curriculum for deterministic evaluation | 强制关闭课程学习
curriculum = false

# Standard mode: deterministic sub-terrain list (no proportion)
# Each terrain occupies one column; env_i → (level[(i//n_sub)%n_lv], sub[i%n_sub])
# Standard 模式子地形列表, 每种地形占一列, env 按笛卡尔积依次放置
[terrain.standard]
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv"]

# Track mode: linear course config (used only when mode = "track")
# Track 模式赛道配置(仅当 mode = "track" 时生效)
[terrain.track]
# Number of sub-terrains chained along the track | 赛道串联的子地形数量
track_length = 5
# Sub-terrain sequence along the track (length must equal track_length, no duplicates)
# Allowed: pyramid_slope | pyramid_slope_inv | pyramid_stairs | pyramid_stairs_inv | open_entry_maze
# 赛道上子地形顺序(长度=track_length, 不可重复)
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
# Parallel tracks along Y-axis | Y 轴并行赛道数
# Default: 10. Each robot is placed on the track whose column index matches
# its assigned level from `[terrain] level = [...]`. The framework only
# auto-extends when max(level)+1 > 10.
# 默认 10 条并行赛道; 每只机器人按 [terrain] level 列表分配到对应 col 的赛道上;
# 仅当 max(level)+1 > 10 时框架才自动扩展 num_parallel_tracks。
num_parallel_tracks = 10

# Velocity command limits | 速度指令上限
[commands]

[commands.limit]
# Linear velocity X range (m/s) | X 方向线速度范围
lin_vel_x = [0.0, 1.0]
# Linear velocity Y range (m/s) | Y 方向线速度范围
lin_vel_y = [-0.0, 0.0]
# Angular velocity Z range (rad/s) | Z 轴角速度范围
ang_vel_z = [0.0, 0.0]
```


⚠️ 注意

评估时部分训练配置（如域随机化、观测噪声）默认关闭，以确保评估结果的一致性和可复现性。
task 必须与 mode 保持一致；若不一致，本轮评估结果没有意义（评分字段与地形语义错位）。
num_envs 配置小于等于 16 时会产出 mp4 视频用于可视化观察模型表现，大于 16 时将不会产出 mp4 视频。

# 数据协议
为了方便同学们调用原始数据和特征数据，下面提供了协议供大家查阅。

注意：基于 Isaac Lab 仿真环境，所有观测和动作数据均以 torch.Tensor 格式在 GPU 上传递，而非传统的字典协议格式。

## 环境交互接口
### 环境重置（Reset）
```python
# 环境重置
reset_data = env.reset(usr_conf)

# 成功时
obs, critic_obs = reset_data
# obs: torch.Tensor, shape=(num_envs, obs_dim)       — policy 观测
# critic_obs: torch.Tensor, shape=(num_envs, critic_obs_dim) — critic 观测

# 失败时
reset_data = None
```

### 环境步进（Step）
```python
data = env.step(actions)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data
dones = terminated | truncated
```

### reset() 返回结构：

字段名	字段路径	类型	取值范围	说明
obs	reset()[0]	torch.Tensor	(num_envs, obs_dim)	policy 观测
critic_obs	reset()[1]	torch.Tensor	(num_envs, critic_obs_dim)	critic 观测
### step() 返回结构：

字段名	字段路径	类型	取值范围	说明
frame_no	step()[0]	int	>=1	当前交互帧号
obs	step()[1]	torch.Tensor	(num_envs, obs_dim)	下一步 policy 观测
rewards	step()[2]	torch.Tensor	(num_envs,)	当前步总 reward
terminated	step()[3]	torch.Tensor[bool]	(num_envs,)	真实终止
truncated	step()[4]	torch.Tensor[bool]	(num_envs,)	超时截断
infos	step()[5][0]	dict	—	Isaac Lab extras
privileged_obs	step()[5][1]	torch.Tensor	(num_envs, critic_obs_dim)	critic 观测
## 观测数据协议（Observation）
### 策略观测（Policy Observation）
obs = [proprio(45) | height_scan(256)]  →  301 维

#### proprio（45 维）字段布局
区间	维度	含义	来源
[0:3]	3	base_ang_vel，机体角速度，scale=0.25	Isaac Lab mdp
[3:6]	3	projected_gravity，重力方向投影	Isaac Lab mdp
[6:9]	3	velocity_commands (vx, vy, wz)	command manager
[9:21]	12	joint_pos_rel，关节相对默认位置	robot data
[21:33]	12	joint_vel_rel，关节速度，scale=0.05	robot data
[33:45]	12	last_action，上一帧动作	action manager
#### height_scan（256 维）
字段名	字段路径	类型	取值范围	说明
height_scan	obs[:, 45:301]	torch.Tensor	clip [-5, 5]，scale=2.5	16x16 前方高度扫描，256 条射线
#### Track 地形附加原始张量（选手自行消费）
代码包的 default_observation() 返回 301 维 obs，不内置目标点或通行性特征。当 terrain.mode = "track" 时，环境额外暴露以下原始张量，选手需要在 policy_observation_process.py::process() 里自行读取并拼接到 obs，同时同步修改 critic_observation_process.py、模型输入维度和 TOML 里的 obs 维度：

字段	类型	Shape	含义
env.goal_positions	torch.Tensor	(num_envs, 3)	目标点（迷宫出口）在世界坐标系下的 (x, y, z)
env.goal_yaw	torch.Tensor	(num_envs,)	目标点在世界坐标系下的朝向（rad）
env.scene.sensors["nav_scanner"]	RayCaster	—	前瞻遮挡扫描传感器，范围比 height_scanner 更大，适合避障 / 转向判断
这些都是原始值；如需构造相关新的特征处理，由选手自行设计与实现。

### 评论家观测（Critic Observation）
critic_obs = [critic_proprio(60) | height_scan(256)]  →  316 维

#### critic_proprio（60 维）字段布局
区间	维度	含义
[0:3]	3	base_lin_vel，机体线速度
[3:6]	3	base_ang_vel，机体角速度
[6:9]	3	projected_gravity
[9:12]	3	velocity_commands
[12:24]	12	joint_pos_rel
[24:36]	12	joint_vel_rel
[36:48]	12	joint_effort，关节力矩
[48:60]	12	last_action
补充说明：critic_proprio（60 维）相比 proprio（45 维）额外包含 base_lin_vel 和 joint_effort 等特权数据，仅供 Critic 网络训练使用，体现"不对称 Actor-Critic"设计。

## 动作数据协议（Action）
actions = torch.Tensor  # shape: (num_envs, 12)

字段名	类型	Shape	取值范围	说明
actions	float32	(num_envs, 12)	[-1.0, 1.0]	12 个关节控制动作
### 关节动作维度
Go2 为 12 自由度四足机器人，4 条腿 x 3 个关节：

维度	关节组	说明
0~2	前左腿	hip / thigh / calf
3~5	前右腿	hip / thigh / calf
6~8	后左腿	hip / thigh / calf
9~11	后右腿	hip / thigh / calf
注意：动作值为归一化偏移量，经 action_scale（默认 0.25）缩放后加到默认关节角度上，作为 PD 控制器目标角度。

## 奖励数据协议（Reward）
字段名	类型	Shape	说明
rewards	float32	(num_envs,)	当前步总 reward
### 代码包默认激活的奖励项
以下 reward 已在 train_env_conf_standard_locomotion.toml 的 [rewards.*] 段激活：

奖励名称	作用	类型
track_lin_vel_xy	奖励 XY 速度跟踪	正奖励
track_ang_vel_z	奖励 yaw 角速度跟踪	正奖励
lin_vel_z	惩罚 Z 方向跳动	负奖励
ang_vel_xy	惩罚 roll/pitch 角速度	负奖励
joint_acc	惩罚关节加速度	负奖励
joint_torques	惩罚关节扭矩	负奖励
dof_pos_limits	惩罚接近关节极限	负奖励
action_rate	惩罚动作帧间变化率	负奖励
undesired_contacts	惩罚非脚掌接触	负奖励
flat_orientation	惩罚非直立姿态	负奖励
reach_goal	到达目标点奖励（需 env.goal_positions 有效，默认 track 地形生效）	正奖励
其它框架内置 reward（如 energy、feet_stumble、feet_height_body、termination 等）未默认激活，选手若需启用需在 reward_process.py 实现并在 TOML 里新增 [rewards.<name>] 段。

## 终止与超时协议
字段	类型	Shape	说明
terminated	bool	(num_envs,)	真实终止
truncated	bool	(num_envs,)	超时截断
条件	terminated	truncated	说明
episode 最大时长	False	True	超时截断
姿态异常/摔倒	True	False	真实失败
Track 到达出口	True	False	导航成功
算法侧应使用 dones = terminated | truncated 判断环境是否结束。

## 访问示例
### 解析 step() 返回
```python
def parse_env_step(data):
    if data is None:
        raise RuntimeError("env.step failed")
    frame_no, obs, rewards, terminated, truncated, extra = data
    infos, privileged_obs = extra
    critic_obs = privileged_obs if privileged_obs is not None else obs
    dones = torch.logical_or(terminated, truncated)
    return {
        "frame_no": frame_no,
        "obs": obs,
        "critic_obs": critic_obs,
        "rewards": rewards,
        "dones": dones,
        "terminated": terminated,
        "truncated": truncated,
        "infos": infos,
    }
```

### 解析 policy observation
```python
def parse_policy_obs(obs: torch.Tensor):
    return {
        "base_ang_vel": obs[:, 0:3],
        "projected_gravity": obs[:, 3:6],
        "velocity_commands": obs[:, 6:9],
        "joint_pos_rel": obs[:, 9:21],
        "joint_vel_rel": obs[:, 21:33],
        "last_action": obs[:, 33:45],
        "height_scan": obs[:, 45:301],
    }
```

## 常见问题
Q1: Critic 观测和 Policy 观测的区别？
Critic 观测包含 critic_proprio（60 维），额外含 base_lin_vel 和 joint_effort 等特权信息，仅训练时可用，体现不对称 Actor-Critic 设计。

Q2: 返回格式与传统环境的区别？
reset() 返回 (obs, critic_obs) 而非字典。step() 返回 (frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs))，需用 dones = terminated | truncated 合并。
