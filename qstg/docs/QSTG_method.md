方法：针对弱监督视频时序定位中“候选提议包含与查询相关的局部线索，却未必包含一个关系正确、时间连贯的完整事件”的问题，本文提出查询子图与时间证据图联合建模模块 Query-Subgraph Temporal Grounder（QSTG）。给定视频及其文本查询，定位器先生成 $K$ 个候选提议，用候选内的视频特征重构查询，再根据文本重构误差获得弱监督排序信号。其中，$K$ 是候选提议总数。第 $k$ 个候选的正提议时间掩码记为 $\mathbf{m}_k^p$，预测区间记为 $\mathbf{P}_k=[l_k^p,r_k^p]$，文本重构误差记为 $L_{ce}^p[k]$。其中，$l_k^p$ 和 $r_k^p$ 分别是归一化的预测起点与终点。候选的多高斯形式为

$$
\mathbf{m}_k^p=\sum_{j=1}^{J_k}\pi_{k,j}\mathbf{g}_{k,j}^p,
\qquad \sum_{j=1}^{J_k}\pi_{k,j}=1,
\qquad \mathbf{g}_{k,j}^p\in\mathbb R^N.
$$

其中，$N$ 是采样后的视频时间位置数，$J_k$ 是第 $k$ 个候选的高斯子提议数，$\mathbf{g}_{k,j}^p$ 是第 $j$ 个子提议在 $N$ 个时间位置上的掩码向量，$\pi_{k,j}$ 是该子提议的重要性权重。QSTG 进一步判断查询中的必要语义是否由同一候选内部的视觉证据共同支持，以及这些证据的时序关系是否成立。

具体地，QSTG 从模型实际接收的查询词序列中抽取动作、实体和属性短语，并根据动作与参与实体、属性与实体、动作先后及同时关系建立查询子图。对第 $\nu$ 个短语，先聚合其包含的有效词，再沿查询子图的边传播信息，得到短语表示：

$$
\mathbf{u}_\nu^{(0)}=
\frac{\sum_{\ell}M_{\nu,\ell}\mathbf{x}_\ell}
{\sum_{\ell}M_{\nu,\ell}+\varepsilon}
+\mathbf{d}_{\operatorname{type}(\nu)},
\qquad
\mathbf{u}_\nu=\operatorname{GraphEnc}(\mathbf{u}_\nu^{(0)}).
$$

其中，$\mathbf{x}_\ell$ 是查询中第 $\ell$ 个有效词的上下文特征向量，$M_{\nu,\ell}$ 是该词属于短语 $\nu$ 的标量指示量，$\operatorname{type}(\nu)$ 是短语类型，$\mathbf{d}_{\operatorname{type}(\nu)}$ 是相应的类型嵌入向量，$\varepsilon$ 是避免分母为零的正常数，$\mathbf{u}_\nu^{(0)}$ 和 $\mathbf{u}_\nu$ 分别是传播前后的短语表示向量，$\operatorname{GraphEnc}$ 是仅沿有效查询图边聚合信息的编码器。将必须获得视觉支持的内容短语组成集合 $\mathcal R$，将短语间的有效关系组成集合 $\mathcal G_{\mathrm{rel}}$。其中，$\mathcal R$ 是候选必须覆盖的短语集合，$\mathcal G_{\mathrm{rel}}$ 是带有关系类型和方向的短语边集合。若过滤和截断后没有有效内容短语，则用一个全局查询节点作为 $\mathcal R$ 中的唯一节点，以保持查询仍可获得视觉支持。

视频侧将逐时间位置特征 $\mathbf{h}_i$ 聚合为时间证据节点。其中，$\mathbf{h}_i$ 是视频在第 $i$ 个采样位置的视觉特征向量。相邻位置的池化方式为

$$
\mathbf{v}_t=\frac{1}{|\mathcal I_t|}\sum_{i\in\mathcal I_t}\mathbf{h}_i,
\qquad t\in\mathcal T.
$$

其中，$\mathcal I_t$ 是第 $t$ 个时间窗口所包含的视频位置集合，$\mathbf{v}_t$ 是该窗口的视觉特征向量，$\mathcal T$ 是有效时间节点集合。为保留时间顺序，QSTG 先连接节点自身及相邻时间节点，再从有限时间范围内选择视觉特征相似的节点建立补充边。时间图的邻接矩阵定义为

$$
\mathbf{A}^{\mathrm{time}}=
\bigl[A^{\mathrm{time}}_{t,t'}\bigr]_{t,t'\in\mathcal T},
\qquad
A^{\mathrm{time}}_{t,t'}=
\mathbb{1}\bigl[t=t'\ \lor\ |t-t'|=1\ \lor\ t'\in\mathcal S_t\bigr].
$$

其中，$\mathbf{A}^{\mathrm{time}}$ 是时间图的邻接矩阵，$A^{\mathrm{time}}_{t,t'}\in\{0,1\}$ 是节点 $t$ 到节点 $t'$ 是否存在图边的标量指示量，$\mathcal S_t$ 是节点 $t$ 在限定时间距离内、视觉余弦相似度达到阈值的至多 $k_{sim}$ 个最相似节点构成的集合，$k_{sim}$ 是每个节点可增加的视觉相似边数。时间图只由视频特征建立，因此在批内构造错误视频与查询的配对时，不会将正确查询的信息预先写入被比较的视频节点。

随后，QSTG 将短语节点和时间节点投影到同一空间，建立多对多的软绑定：

$$
s_{\nu,t}=\left\langle
\operatorname{norm}(\mathbf{W}_q\mathbf{u}_\nu),
\operatorname{norm}(\mathbf{W}_v\mathbf{v}_t)
\right\rangle,
\qquad
B_{\nu,t}=\frac{\exp(s_{\nu,t}/\tau_b)}
{\sum_{t'\in\mathcal T}\exp(s_{\nu,t'}/\tau_b)}.
$$

其中，$s_{\nu,t}$ 是短语 $\nu$ 与时间节点 $t$ 的标量余弦相似度，$\mathbf{W}_q$ 和 $\mathbf{W}_v$ 是可学习的文本与视频投影矩阵，$\operatorname{norm}$ 是单位长度归一化操作，$B_{\nu,t}$ 是短语在时间节点上的标量绑定权重，$\tau_b$ 是绑定温度。该绑定允许同一短语由多个时间节点支持，也允许同一节点支持多个短语。为了衡量每个时间节点与查询的相关性，对必须覆盖的短语作平滑聚合：

$$
\eta_t=\sigma\!\left[
\tau_r\log\left(
\frac{1}{|\mathcal R|}
\sum_{\nu\in\mathcal R}\exp(s_{\nu,t}/\tau_r)
\right)\right].
$$

其中，$\eta_t$ 是时间节点 $t$ 的查询相关性，$\tau_r$ 是短语聚合温度，$\sigma$ 是 Sigmoid 函数。单个节点的高相关性只能说明存在局部语义线索，尚不能说明查询中每个必要动作和实体均已在同一候选内获得支持。

为使动作先后和同时关系参与定位，QSTG 为查询关系边 $(\nu,\mu,\xi)\in\mathcal G_{\mathrm{rel}}$ 定义有方向的时间关系核：

$$
\kappa_\xi(t,t')=
\begin{cases}
\mathbb{1}[t'\ge t]\exp(-|t-t'|/\lambda_{rel}), & \xi=\mathrm{before},\\
\mathbb{1}[t'\le t]\exp(-|t-t'|/\lambda_{rel}), & \xi=\mathrm{after},\\
\exp(-|t-t'|/\lambda_{sim}), & \xi=\mathrm{while},\\
\exp(-|t-t'|/\lambda_{rel}), & \xi=\mathrm{assoc},
\end{cases}
\qquad \lambda_{sim}<\lambda_{rel}.
$$

其中，$\nu$ 和 $\mu$ 分别是关系边的起点与终点短语，$\xi$ 是先于、后于、同时或一般关联的关系类型，$\kappa_\xi(t,t')$ 是时间节点对满足该关系的标量程度，$\mathbb{1}[\cdot]$ 是取值为 0 或 1 的条件指示函数，$\lambda_{sim}$ 和 $\lambda_{rel}$ 分别是同时关系与其他关系的时间衰减尺度。查询关系与软绑定共同形成时间图上的关系偏置：

$$
\Gamma(t,t')=A^{\mathrm{time}}_{t,t'}
\frac{1}{|\mathcal G_{\mathrm{rel}}|}
\sum_{(\nu,\mu,\xi)\in\mathcal G_{\mathrm{rel}}}
B_{\nu,t}B_{\mu,t'}\kappa_\xi(t,t').
$$

其中，$\Gamma(t,t')$ 是已有时间图边上的标量查询关系偏置。当 $\mathcal G_{\mathrm{rel}}$ 为空时令 $\Gamma(t,t')=0$。时间图传播结合 $A^{\mathrm{time}}_{t,t'}$、$\eta_t$ 与 $\Gamma(t,t')$ 得到更新的节点特征 $\widetilde{\mathbf{v}}_t$。其中，$\widetilde{\mathbf{v}}_t$ 是融合查询结构后的时间证据向量。关系偏置只调节原有图边，不因查询相似度直接连接相距很远的片段。

对于第 $k$ 个候选，先将正提议掩码映射到时间图窗口，再计算必要短语覆盖度和候选内部相关性：

$$
A_{k,t}=\frac{1}{|\mathcal I_t|}
\sum_{i\in\mathcal I_t}m_{k,i}^p,
\qquad
C_k=\frac{1}{|\mathcal R|}
\sum_{\nu\in\mathcal R}\sum_{t\in\mathcal T}A_{k,t}B_{\nu,t},
$$

$$
\widetilde A_{k,t}=A_{k,t}\mathbb{1}[A_{k,t}\ge\delta_A],
\qquad
X_k=\frac{\sum_{t\in\mathcal T}\widetilde A_{k,t}\eta_t}
{\sum_{t\in\mathcal T}\widetilde A_{k,t}+\varepsilon}.
$$

其中，$m_{k,i}^p$ 是 $\mathbf{m}_k^p$ 在时间位置 $i$ 的标量分量，$A_{k,t}$ 是候选对节点 $t$ 的标量软覆盖权重，$C_k$ 是必要短语的平均证据覆盖度，$\widetilde A_{k,t}$ 是去除低权重掩码尾部后的覆盖值，$\delta_A$ 是尾部过滤阈值，$X_k$ 是候选主要覆盖位置的平均查询相关性。$C_k$ 检验必要语义是否进入候选；$X_k$ 则避免候选为覆盖少量相关节点而同时包含大段无关背景。

仅有短语覆盖仍可能把时间顺序错误的线索误判为目标事件。因此，在候选覆盖的时间节点对上计算关系满足度：

$$
T_k^{\nu\mu}(t,t')=
A_{k,t}A_{k,t'}B_{\nu,t}B_{\mu,t'}A^{\mathrm{time}}_{t,t'},
$$

$$
R_k=
\frac{\sum_{(\nu,\mu,\xi)\in\mathcal G_{\mathrm{rel}}}\sum_{t,t'\in\mathcal T}
T_k^{\nu\mu}(t,t')\kappa_\xi(t,t')}
{\sum_{(\nu,\mu,\xi)\in\mathcal G_{\mathrm{rel}}}\sum_{t,t'\in\mathcal T}
T_k^{\nu\mu}(t,t')+\varepsilon}.
$$

其中，$T_k^{\nu\mu}(t,t')$ 是候选内部短语对和时间节点对的联合证据权重，$R_k$ 是候选内有效关系的平均满足度。若查询没有有效关系边，则令 $R_k=0$，并设置关系有效指示量 $I^{rel}=0$；若有关系边但候选内没有可比较的节点对，$R_k$ 同样取 0。其中，$I^{rel}$ 是查询是否含有效关系边的指示量，有效时取 1。这样，缺少关系信息不会被当作关系完全成立。

同时，QSTG 估计候选是否跨越了缺少语义支持的明显证据断点：

$$
b_t=\bigl(1-\cos(\mathbf{v}_t,\mathbf{v}_{t+1})\bigr)
|\eta_t-\eta_{t+1}|\bigl(1-\Gamma(t,t+1)\bigr),
$$

$$
H_k=\exp\!\left[-
\frac{\sum_{t:\,t,t+1\in\mathcal T}A_{k,t}A_{k,t+1}b_t}
{\sum_{t:\,t,t+1\in\mathcal T}A_{k,t}A_{k,t+1}+\varepsilon}
\right].
$$

其中，$b_t$ 是相邻时间节点间缺少关系支持的视觉与语义变化量，$\cos$ 是余弦相似度函数，$H_k$ 是候选内部时间证据的连贯性。低 $H_k$ 表明候选可能跨越明显断点；高 $H_k$ 需要结合短语覆盖和关系满足度判断，不能单独解释为边界正确。将以上证据连同预测区间的几何信息送入候选质量函数：

$$
w_k=r_k^p-l_k^p,
\qquad
D_k=\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}
|A_{k,t}-A_{k,t}^{box}|,
\qquad
y_k=f_{qual}(C_k,X_k,I^{rel}R_k,H_k,w_k,D_k).
$$

其中，$w_k$ 是预测区间 $\mathbf{P}_k$ 的实际宽度，$A_{k,t}^{box}$ 是由该区间构造的时间窗口标量软覆盖，$D_k$ 是正提议掩码与预测区间的覆盖差异，$f_{qual}$ 是可学习的质量评分函数，$y_k$ 是第 $k$ 个候选的标量结构性质量分数。区间宽度始终由 $r_k^p-l_k^p$ 得到，不以掩码面积代替。

由于训练阶段没有候选级真实 IoU，QSTG 使用视频级配对关系学习短语绑定，并用文本重构误差学习候选排序。对于批内查询 $b$ 与视频 $c$，先用不含查询条件的视频节点聚合短语相似度：

$$
S_{b,c}=\frac{1}{|\mathcal R_b|}
\sum_{\nu\in\mathcal R_b}
\tau_m\log\left[
\frac{1}{|\mathcal T_c|}
\sum_{t\in\mathcal T_c}
\exp\!\left(\frac{s_{b,c,\nu,t}}{\tau_m}\right)
\right],
$$

$$
L_{mc}^{q\to v}=-\frac{1}{|\mathcal B|}
\sum_{b\in\mathcal B}
\log\frac{\exp(S_{b,b}/\tau_c)}
{\sum_{c\in\{b\}\cup\mathcal N(b)}\exp(S_{b,c}/\tau_c)}.
$$

其中，$b$ 是查询及其正确配对视频在批内的索引，$c$ 是被比较视频的索引，$\mathcal R_b$ 和 $\mathcal T_c$ 分别是查询 $b$ 的必要短语集合与视频 $c$ 的有效时间节点集合，$s_{b,c,\nu,t}$ 是该查询—视频组合的短语节点相似度，$\tau_m$ 是节点分数聚合温度，$S_{b,c}$ 是视频级匹配分数，$\mathcal B$ 是具有有效异视频负例的查询集合，$\mathcal N(b)$ 是批内与查询 $b$ 所属视频不同的视频索引集合，$\tau_c$ 是对比温度，$L_{mc}^{q\to v}$ 是查询到视频方向的匹配损失。交换查询和视频得到 $L_{mc}^{v\to q}$，并令 $L_{mc}=(L_{mc}^{q\to v}+L_{mc}^{v\to q})/2$。其中，$L_{mc}^{v\to q}$ 是视频到查询方向的匹配损失，$L_{mc}$ 是双向视频级匹配损失。同一视频的其他查询不作为负例。

候选质量分支依据文本重构误差构造软选择权重 $a_k$，使结构性质量排序与重构提供的弱排序信号一致：

$$
a_k=\operatorname{softmax}_k\left(-\frac{L_{ce}^p[k]}{\tau_e}\right),
\qquad
L_{pq}=-\sum_{k=1}^{K}\operatorname{stopgrad}(a_k)
\log\frac{\exp(y_k/\tau_q)}
{\sum_{j=1}^{K}\exp(y_j/\tau_q)}.
$$

其中，$a_k$ 是第 $k$ 个候选的重构软选择权重，$\tau_e$ 是候选选择温度，$L_{pq}$ 是候选质量排序损失，$\tau_q$ 是质量排序温度，$\operatorname{stopgrad}$ 是停止梯度操作。该损失只提供重构式弱监督，不把重构误差或质量分数解释成真实边界标签。训练初期固定候选生成器，先学习短语绑定和质量评分；待绑定具有区分度后，再将图信息以残差形式反馈给候选生成器：

$$
\mathbf{v}^Q=\frac{\sum_{t\in\mathcal T}\eta_t\widetilde{\mathbf{v}}_t}
{\sum_{t\in\mathcal T}\eta_t+\varepsilon},
\qquad
\widehat{\mathbf{h}}^{gen}=\mathbf{h}^{gen}+\chi F_\theta([\mathbf{h}^{gen};\mathbf{v}^Q]).
$$

其中，$\mathbf{v}^Q$ 是查询相关的时间图摘要向量，$\mathbf{h}^{gen}$ 是候选生成器的输入摘要向量，$\widehat{\mathbf{h}}^{gen}$ 是加入图信息后的摘要向量，$F_\theta$ 是输出层零初始化的向量值残差映射，$\chi$ 是分阶段启用的图反馈系数。训练初期令 $\chi=0$，使候选掩码与区间在固定生成器条件下保持不变；图反馈开启后，上下文、重叠和多高斯多样性约束继续限制候选扩张。

最终，QSTG 的新增目标与重构式定位目标联合优化：

$$
L_{QSTG}=\lambda_{mc}L_{mc}+\lambda_{pq}L_{pq},
$$

$$
L=L_{rec}+\alpha_1L_{IVC}+\alpha_2L_{div}
+L_{event}+L_{mix}+L_{QSTG}.
$$

其中，$L_{QSTG}$ 是 QSTG 的新增损失，$\lambda_{mc}$ 和 $\lambda_{pq}$ 分别是视频级匹配与候选质量排序项的权重，$L_{rec}$ 是文本重构损失，$L_{IVC}$ 是正提议与上下文负提议的排序损失，$L_{div}$ 是候选差异约束，$L_{event}$ 是事件语义分离与边界约束，$L_{mix}$ 是多高斯子提议结构约束，$\alpha_1$ 和 $\alpha_2$ 分别控制排序与候选差异损失的权重。

推理时，QSTG 将短语覆盖、候选内部相关性、有效关系和时间连贯性合成结构分数，并与文本重构误差及学习到的质量分数一起排序：

$$
\psi_k=
\frac{\omega_C C_k+\omega_X X_k+I^{rel}\omega_R R_k+\omega_H H_k}
{\omega_C+\omega_X+I^{rel}\omega_R+\omega_H},
\qquad
F_k=-L_{ce}^p[k]+\lambda_y\sigma(y_k)+\lambda_\psi\psi_k.
$$

其中，$\psi_k$ 是候选 $k$ 的解析结构分数，$\omega_C$、$\omega_X$、$\omega_R$ 和 $\omega_H$ 分别是覆盖、内部相关性、关系和连贯性项的非负权重，$F_k$ 是最终候选排序分数，$\lambda_y$ 和 $\lambda_\psi$ 分别是质量分数与解析结构分数的融合权重。查询没有有效关系边时，关系项及其权重同时从 $\psi_k$ 中移除。最终按 $F_k$ 排序输出 Rank-1 候选及得分靠前的五个候选；输出区间仍为 $\mathbf{P}_k$，所有评分权重仅由验证集确定。
