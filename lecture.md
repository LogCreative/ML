
### 最小平均距离

$$\frac{\partial f}{\partial x}
= \begin{bmatrix}\partial x_1 \\ \partial x_2 \\ \vdots \\ \partial x_n\end{bmatrix}$$

$$
f=a^Tx,\quad \frac{\partial f}{\partial x} = a^T
$$

$$
f = x^TAx,\quad \frac{\partial f}{\partial x} =x^T(A+A^T)
$$

$$
\begin{align*}
f(x) &= \sum_{i=1}^N ||\mu-x_i||^2\\
&=\sum_{i=1}^N (\mu-x_i)^T(\mu-x_i)\\
&=\sum_{i=1}^N (\mu^T\mu - 2x_i^T\mu + x_i^Tx_i)\\
\frac{\partial f}{\partial \mu} &= \sum_{i=1}^N(2\mu^T-2x_i^T)=0\\
\Rightarrow \mu &= \frac{1}{N}\sum_{i=1}^Nx_i
\end{align*}
$$

可以用来推导数：

$$
\begin{align*}
\textrm{Tr}(A) &= \textrm{Tr}(A^T)\\
\textrm{Tr}(ABC) &= \textrm{Tr}(CAB)
\end{align*}
$$

$$
d\text{Tr}[f(x)] = \text{Tr}\left[\left(\frac{\partial f}{\partial x}\right)^Tdx\right]
$$

### 聚类

$$
\text{minimize }J=\sum_{n=1}^N\sum_{k=1}^Nr_{nk}||x_n-\mu_k||^2\\
\text{where }r_{nk} =\begin{cases}
1,\text{if }x_n\text{ is assigned to cluster }k\\
0,\text{otherwise}
\end{cases}
$$

### GMM

$$
G(\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma}) = (2\pi)^{-\frac{d}{2}}|\Sigma|^{-\frac{1}{2}}e^{-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})}
$$

$$
\frac{\partial \ln p(\mathbf{X}|\mathbf{\mu},\mathbf{\Sigma})}{\partial \mathbf{\mathbf{\mu}}}=0 \Rightarrow \mathbf{\mu}_\text{ML}=\frac{1}{N}\sum_{n=1}^N\mathbf{x}_n
$$

$$
\frac{\partial \ln p(\mathbf{X}|\mathbf{\mu},\mathbf{\Sigma})}{\partial \mathbf
\Sigma}=0\Rightarrow \Sigma_\text{ML}=\frac{1}{N}\sum_{n=1}^N(x_n-\mu)(x_n-\mu)^T
$$

### EM

$p(x)\leftrightarrow q(x|\theta)$

$$
\begin{aligned}
KL(p||q)&=\min_{\theta}\int p(x)\log \frac{p(x)}{q(x|\theta)}
\\&=\min_{\theta}\int p(x)\log p(x) - \int p(x)\log q(x|\theta)
\\&=\min_{\theta}\int \frac{1}{N}\sum_{t=1}^N \delta(x-x_t)\log q(x|\theta)
\\&=\frac{1}{N}\sum_{t=1}^N\log q(x_t|\theta) && \int \delta(x)f(x) = f(0)
\end{aligned}$$

$p(x)p(y|x)\leftrightarrow q(x|y,\theta)q(y|\theta)$

$$
\begin{aligned}
\max F(p(y|x),\theta)&=\int p(x)p(y|x)\log\frac{q(x|y,\theta)q(y|\theta)}{p(y|x)}\\
&=\int p(x)p(y|x)\log\frac{q(x|y,\theta)q(y|\theta)}{q(x|\theta)}\frac{q(x|\theta)}{p(y|x)}\\
&=\int p(x)p(y|x)\left[\log q(x|\theta)-\log\frac{p(y|x)}{q(y|x,\theta)} \right]\\
&=\frac{1}{N}\sum_{t=1}^N\log q(x_t|\theta) - \int p(x)\cdot KL(p(y|x)||q(y|x,\theta))\\
&\leq\log q(x|\theta) @ [p(y|x)=q(y|x,\theta)]
\end{aligned}$$

(Free Energy)

fix $p(y|x)$, $F=\int p(y|x)\log[q(x|y,\theta)q(y|\theta)]$

E Step - fix $\theta: p(y|x)=q(y|x,\theta)$

$p(x)p(y|x,\theta)p(\theta|x)\leftrightarrow q(x|y,\theta)q(y|\theta)q(\theta)$

$$
\begin{aligned}
F&=\int p(x)p(y|x,\theta)p(\theta|x)\log\frac{q(x|y,\theta)q(y|\theta)q(\theta)}{p(y|x,\theta)p(\theta|x)}\\
&=\int p(x)p(y|x,\theta)p(\theta|x)\log\frac{q(x|y,\theta)q(y|\theta)}{q(x|\theta)}\frac{q(x|\theta)q(\theta)}{p(y|x,\theta)p(\theta|x)}\\
&=\int p(x)p(y|x,\theta)p(\theta|x)\log\frac{q(y|x,\theta)}{p(y|x,\theta)}\frac{q(x|\theta)q(\theta)}{q(x|k)p(\theta|x)}q(x|k)\\
&=\int p(x)p(y|x,\theta)p(\theta|x)\left(\log\frac{q(y|x,\theta)}{p(y|x,\theta)}+\log\frac{q(\theta|x)}{p(\theta|x)}+\log q(x|k)\right) & \text{VBEM}\\
\end{aligned}
$$

$\max_k q(x|k)=\int q(x|\theta)q(\theta)d\theta$ marginal likelihood

ML $\log q(x|\theta)$

BL $\log q(x|\theta)q(\theta)=\log q(x|\theta) + \log q(\theta)$ 后项为正则化项

