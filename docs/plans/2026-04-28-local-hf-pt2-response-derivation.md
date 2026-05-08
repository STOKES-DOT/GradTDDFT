# 局域 HF/PT2 混合项的严格响应推导（2026-04-28）

**状态**: 理论稿  
**目标**: 把当前 neural local-mixing 结构中的 HF 与 PT2 部分写成统一的二阶响应形式，并明确区分：

1. 严格理论上应该保留的项
2. 当前代码里实际保留的近似项
3. 后续如果要升级 strict response，应当增加哪些响应通道

**当前实现相关文件**

- [src/td_graddft/neural_xc/dm21/functional.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/neural_xc/dm21/functional.py)
- [src/td_graddft/reference.py](/Users/jiaoyuan/Documents/GitHub/TD-GradDFT/src/td_graddft/reference.py)

---

## 1. 总体设定

记局域密度相关变量为

\[
u(\mathbf r)=\big(u_\mu(\mathbf r)\big)
\]

对当前 restricted MGGA 路径，可理解为

\[
u=(\rho,\partial_x\rho,\partial_y\rho,\partial_z\rho,\tau)
\]

若后续启用 QAC/laplacian，再把 \(\nabla^2\rho\) 加入即可。

定义两个局域混合项：

\[
E_x^{\mathrm{mix}}
=
\int d\mathbf r\; a(u(\mathbf r))\,h_x^{\mathrm{loc}}(\mathbf r)
\]

\[
E_c^{\mathrm{mix}}
=
\int d\mathbf r\; b(u(\mathbf r))\,p_c^{\mathrm{loc}}(\mathbf r)
\]

其中：

- \(a(u)\): 局域 HF mixing coefficient
- \(b(u)\): 局域 PT2 mixing coefficient
- \(h_x^{\mathrm{loc}}(\mathbf r)\): 选定 gauge 下的局域 HF 交换能量密度
- \(p_c^{\mathrm{loc}}(\mathbf r)\): 选定 gauge 下的局域 PT2 相关能量密度

若用轨道旋转参数 \(\kappa_{ia}\) 做响应，记复合指标

\[
I\equiv(ia),\qquad J\equiv(jb)
\]

则我们要的 kernel 元素是

\[
K_{IJ}
=
\frac{\partial^2 E}{\partial \kappa_I \partial \kappa_J}
\]

下文分别讨论 HF 与 PT2。

---

## 2. 局域 HF 混合项的严格二阶响应

### 2.1 分解形式

对

\[
E_x^{\mathrm{mix}}
=
\int d\mathbf r\; a(u(\mathbf r))\,h_x^{\mathrm{loc}}(\mathbf r)
\]

做二阶导，严格应分成三块：

\[
K^{x,\mathrm{mix}}_{IJ}
=
K^{aa}_{IJ}
+
K^{ah}_{IJ}
+
K^{hh}_{IJ}
\]

其中：

\[
K^{aa}_{IJ}
=
\int d\mathbf r\;
h_x^{\mathrm{loc}}(\mathbf r)
\sum_{\mu\nu}
a_{,\mu\nu}(\mathbf r)\,
\delta u_\mu^I(\mathbf r)\,
\delta u_\nu^J(\mathbf r)
\]

\[
K^{ah}_{IJ}
=
\int d\mathbf r\;
\sum_\mu a_{,\mu}(\mathbf r)
\Big[
\delta u_\mu^I(\mathbf r)\,\delta h_x^J(\mathbf r)
+
\delta u_\mu^J(\mathbf r)\,\delta h_x^I(\mathbf r)
\Big]
\]

\[
K^{hh}_{IJ}
=
\int d\mathbf r\;
a(\mathbf r)\,\delta^2 h_x^{IJ}(\mathbf r)
\]

记号约定：

\[
a_{,\mu}=\frac{\partial a}{\partial u_\mu},\qquad
a_{,\mu\nu}=\frac{\partial^2 a}{\partial u_\mu\partial u_\nu}
\]

\[
\delta u_\mu^I(\mathbf r)
=
\frac{\partial u_\mu(\mathbf r)}{\partial \kappa_I}
\]

\[
\delta h_x^I(\mathbf r)
=
\frac{\partial h_x^{\mathrm{loc}}(\mathbf r)}{\partial \kappa_I}
\]

\[
\delta^2 h_x^{IJ}(\mathbf r)
=
\frac{\partial^2 h_x^{\mathrm{loc}}(\mathbf r)}
{\partial \kappa_I\partial \kappa_J}
\]

解释如下：

- \(K^{aa}\): 前面的系数按 meta-GGA 类型对局域变量做二阶导
- \(K^{ah}\): 系数导数与局域 HF 能量密度的一阶轨道导数耦合
- \(K^{hh}\): 局域 HF 能量密度本身按传统 HF 逻辑做二阶导

### 2.2 exchange-hole gauge 下的 \(h_x^{\mathrm{loc}}\)

若采用 exchange-hole gauge，可写为

\[
h_{x,\sigma}^{\mathrm{loc}}(\mathbf r)
=
-\frac12
\int d\mathbf r'
\frac{\left|\gamma_\sigma(\mathbf r,\mathbf r')\right|^2}
{|\mathbf r-\mathbf r'|}
\]

其中一体密度矩阵

\[
\gamma_\sigma(\mathbf r,\mathbf r')
=
\sum_{i\in \mathrm{occ},\sigma}
\phi_{i\sigma}(\mathbf r)\phi_{i\sigma}^*(\mathbf r')
\]

总局域交换能量密度为

\[
h_x^{\mathrm{loc}}(\mathbf r)
=
\sum_\sigma h_{x,\sigma}^{\mathrm{loc}}(\mathbf r)
\]

则一阶导数可写为

\[
\delta h_{x,\sigma}^{I}(\mathbf r)
=
-\int d\mathbf r'
\frac{
\gamma_\sigma(\mathbf r,\mathbf r')\,
\delta\gamma_\sigma^{I}(\mathbf r,\mathbf r')
}
{|\mathbf r-\mathbf r'|}
\]

二阶导数可写为

\[
\delta^2 h_{x,\sigma}^{IJ}(\mathbf r)
=
-\int d\mathbf r'
\frac{
\delta\gamma_\sigma^{I}(\mathbf r,\mathbf r')\,
\delta\gamma_\sigma^{J}(\mathbf r,\mathbf r')
}
{|\mathbf r-\mathbf r'|}
\]

对单激发方向 \(I=(ia)\)，常用的一阶 1-RDM 变化写成

\[
\delta\gamma_\sigma^{(ia)}(\mathbf r,\mathbf r')
=
\phi_{a\sigma}(\mathbf r)\phi_{i\sigma}(\mathbf r')
+
\phi_{i\sigma}(\mathbf r)\phi_{a\sigma}(\mathbf r')
\]

### 2.3 常数 HF 混合系数极限

若

\[
a(\mathbf r)=a_x=\mathrm{const.}
\]

则

\[
a_{,\mu}=0,\qquad a_{,\mu\nu}=0
\]

所以

\[
K^{x,\mathrm{mix}}_{IJ}=K^{hh}_{IJ}
\]

并且应退化到标准 hybrid TDDFT/TDA 的 HF 交换核：

\[
A^{HF}_{ia,jb}=-a_x(ij|ab)
\]

\[
B^{HF}_{ia,jb}=-a_x(ib|aj)
\]

这给了 local-mixing HF 路径一个基本一致性检查：

- 常数 mixing 极限必须退化回普通 HF/hybrid kernel

---

## 3. 局域 PT2 混合项的严格二阶响应

### 3.1 分解形式

对

\[
E_c^{\mathrm{mix}}
=
\int d\mathbf r\; b(u(\mathbf r))\,p_c^{\mathrm{loc}}(\mathbf r)
\]

完全平行地做二阶导，可写成

\[
K^{pt2,\mathrm{mix}}_{IJ}
=
K^{bb}_{IJ}
+
K^{bp}_{IJ}
+
K^{pp}_{IJ}
\]

其中

\[
K^{bb}_{IJ}
=
\int d\mathbf r\;
p_c^{\mathrm{loc}}(\mathbf r)
\sum_{\mu\nu}
b_{,\mu\nu}(\mathbf r)\,
\delta u_\mu^I(\mathbf r)\,
\delta u_\nu^J(\mathbf r)
\]

\[
K^{bp}_{IJ}
=
\int d\mathbf r\;
\sum_\mu b_{,\mu}(\mathbf r)
\Big[
\delta u_\mu^I(\mathbf r)\,\delta p_c^J(\mathbf r)
+
\delta u_\mu^J(\mathbf r)\,\delta p_c^I(\mathbf r)
\Big]
\]

\[
K^{pp}_{IJ}
=
\int d\mathbf r\;
b(\mathbf r)\,\delta^2 p_c^{IJ}(\mathbf r)
\]

这里

\[
b_{,\mu}=\frac{\partial b}{\partial u_\mu},\qquad
b_{,\mu\nu}=\frac{\partial^2 b}{\partial u_\mu\partial u_\nu}
\]

\[
\delta p_c^I(\mathbf r)
=
\frac{\partial p_c^{\mathrm{loc}}(\mathbf r)}{\partial \kappa_I}
\]

\[
\delta^2 p_c^{IJ}(\mathbf r)
=
\frac{\partial^2 p_c^{\mathrm{loc}}(\mathbf r)}
{\partial \kappa_I\partial \kappa_J}
\]

解释如下：

- \(K^{bb}\): 前面的 PT2 mixing coefficient 按 meta-GGA 型局域变量做二阶导
- \(K^{bp}\): 系数导数与局域 PT2 能量密度的一阶轨道导数耦合
- \(K^{pp}\): 局域 PT2 能量密度本身的二阶轨道导数

### 3.2 当前代码中的 local exact-pair gauge surrogate

当前实现的 PT2 局域特征可用下式概括：

\[
p_c^{\mathrm{loc}}(\mathbf r)
=
\sum_{iajb}
\rho_{ia}(\mathbf r)\,
V_{jb}(\mathbf r)\,
W_{iajb}
\]

其中

\[
\rho_{ia}(\mathbf r)=\phi_i(\mathbf r)\phi_a(\mathbf r)
\]

\[
W_{iajb}
=
\frac{2(ia|jb)-(ib|ja)}
{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}
\]

\[
V_{jb}(\mathbf r)
=
\sum_{pqrs}
\chi_p(\mathbf r)\chi_q(\mathbf r)
(pq|rs)\,
C_{rj}C_{sb}
\]

这正对应当前 `reference.py` / `functional.py` 中的 local exact-pair gauge surrogate，而不是严格频率依赖的 doubles kernel。

### 3.3 一阶与二阶导数结构

对

\[
p_c^{\mathrm{loc}}(\mathbf r)
=
\sum_{iajb}\rho_{ia}V_{jb}W_{iajb}
\]

一阶导数可写为

\[
\delta p_c^I(\mathbf r)
=
\sum_{iajb}
\Big[
\delta\rho_{ia}^I(\mathbf r)\,V_{jb}(\mathbf r)\,W_{iajb}
+
\rho_{ia}(\mathbf r)\,\delta V_{jb}^I(\mathbf r)\,W_{iajb}
+
\rho_{ia}(\mathbf r)\,V_{jb}(\mathbf r)\,\delta W_{iajb}^I
\Big]
\]

二阶导数则应为

\[
\delta^2 p_c^{IJ}(\mathbf r)
=
\sum_{iajb}
\delta^{IJ}\!\Big(\rho_{ia}V_{jb}W_{iajb}\Big)
\]

展开后包含九类项：

\[
\delta^2 p_c^{IJ}(\mathbf r)
=
\sum_{iajb}
\Big[
\delta^{IJ}\rho_{ia}\,V_{jb}\,W_{iajb}
+
\rho_{ia}\,\delta^{IJ}V_{jb}\,W_{iajb}
+
\rho_{ia}\,V_{jb}\,\delta^{IJ}W_{iajb}
\]
\[
+
\delta^I\rho_{ia}\,\delta^J V_{jb}\,W_{iajb}
+
\delta^J\rho_{ia}\,\delta^I V_{jb}\,W_{iajb}
+
\delta^I\rho_{ia}\,V_{jb}\,\delta^J W_{iajb}
\]
\[
+
\delta^J\rho_{ia}\,V_{jb}\,\delta^I W_{iajb}
+
\rho_{ia}\,\delta^I V_{jb}\,\delta^J W_{iajb}
+
\rho_{ia}\,\delta^J V_{jb}\,\delta^I W_{iajb}
\Big]
\]

这说明如果要“严格仿照 HF 成分做二阶导”，则 PT2 也不能只保留 mixing coefficient 的 Hessian；局域 PT2 能量密度本身的轨道导数必须显式保留。

### 3.4 一个必须明确的理论边界

即使把上面的

\[
K^{bb}+K^{bp}+K^{pp}
\]

都保留，这也仍然只是：

- 对选定的局域 surrogate \(p_c^{\mathrm{loc}}(\mathbf r)\) 的严格二阶导

而**不是**完整的严格 PT2/GL2 频率依赖 doubles kernel。

真正的 PT2 激发态修正一般仍然具有：

- 非局域性
- 频率依赖
- doubles manifold 耦合

所以这里的“严格”应理解为：

- 在选定 local exact-pair gauge surrogate 内部，链式法则不再冻结 \(p_c^{\mathrm{loc}}\)

而不是说它已经等价于 full PT2 TD response。

---

## 4. 当前代码实际保留了什么

### 4.1 当前 strict tensor 的求导对象

当前实现里，strict response tensor 来自

\[
\mathrm{jax.hessian}\big(\_total\_point\_local\_energy\_from\_variables,\ \mathrm{argnums}=1\big)
\]

也就是：

- 只对 `variables` 求 Hessian
- `hf_point` 与 `pt2_point` 都作为单独参数传入

因此，无论 `response_hf_mode="local_projected"` 还是 `response_pt2_mode="local_projected"`，

\[
h_x^{\mathrm{loc}}(\mathbf r)
\]

和

\[
p_c^{\mathrm{loc}}(\mathbf r)
\]

在 Hessian 里都不是响应变量，而是外部常数参数。

### 4.2 因而当前 HF/PT2 direct 项只有系数-Hessian块

当前 local-projected 直接项分别退化成

\[
H^{HF,\mathrm{current}}_{\mu\nu}(\mathbf r)
=
h_x^{\mathrm{loc}}(\mathbf r)\,a_{,\mu\nu}(\mathbf r)
\]

\[
H^{PT2,\mathrm{current}}_{\mu\nu}(\mathbf r)
=
p_c^{\mathrm{loc}}(\mathbf r)\,b_{,\mu\nu}(\mathbf r)
\]

也就是说，当前代码仅保留：

- HF: \(K^{aa}\)
- PT2: \(K^{bb}\)

而没有保留：

- HF: \(K^{ah}\), \(K^{hh}\)
- PT2: \(K^{bp}\), \(K^{pp}\)

### 4.3 与 frozen-field 近似的关系

当前这条路径可以理解成：

- 系数前面的局域 HF / PT2 basis 值保留
- 但它们自身对轨道旋转的变化被冻结

因此当前并不是

\[
\partial_{\kappa_I}\partial_{\kappa_J}\big(a\,h_x^{\mathrm{loc}}\big)
\]

或

\[
\partial_{\kappa_I}\partial_{\kappa_J}\big(b\,p_c^{\mathrm{loc}}\big)
\]

的完整二阶导，而只是各自的第一块：

\[
h_x^{\mathrm{loc}}
\sum_{\mu\nu} a_{,\mu\nu}\,\delta u_\mu^I\,\delta u_\nu^J
\]

\[
p_c^{\mathrm{loc}}
\sum_{\mu\nu} b_{,\mu\nu}\,\delta u_\mu^I\,\delta u_\nu^J
\]

---

## 5. 如果要升级到严格 local-surrogate response，需要补什么

若目标是把 HF/PT2 两块都提升到“在所选局域 surrogate 内部严格”的层级，则最少需要补齐：

1. **HF 响应通道**

\[
\delta h_x^{\mathrm{loc}}(\mathbf r)
\]

2. **PT2 响应通道**

\[
\delta p_c^{\mathrm{loc}}(\mathbf r)
\]

3. 扩展后的响应变量集合

\[
\tilde u(\mathbf r)
=
\big(u(\mathbf r), h_x^{\mathrm{loc}}(\mathbf r), p_c^{\mathrm{loc}}(\mathbf r)\big)
\]

4. 对扩展变量做 Hessian，或者等价地显式实现

- \(K^{aa},K^{ah},K^{hh}\)
- \(K^{bb},K^{bp},K^{pp}\)

否则当前 strict tensor 再精确，也只是对

\[
u(\mathbf r)
\]

这组局域密度变量严格，而对 HF/PT2 surrogate 仍然是 frozen-field 近似。

---

## 6. 结论

对于形如

\[
\int a(u)\,h_x^{\mathrm{loc}}\,d\mathbf r
\]

和

\[
\int b(u)\,p_c^{\mathrm{loc}}\,d\mathbf r
\]

的局域混合项，严格二阶响应都不应只保留“系数像 meta-GGA 那样做 Hessian”的一块，而应统一写成：

\[
\boxed{
K^{x,\mathrm{mix}}=K^{aa}+K^{ah}+K^{hh}
}
\]

\[
\boxed{
K^{pt2,\mathrm{mix}}=K^{bb}+K^{bp}+K^{pp}
}
\]

其中：

- 第一块是系数对局域变量的二阶导
- 第二块是系数导数与局域 HF/PT2 能量密度一阶轨道导数的交叉项
- 第三块是局域 HF/PT2 能量密度本身的二阶轨道导数

而当前代码中的 local-projected direct HF/PT2 响应，仅对应这两式中的第一块。
