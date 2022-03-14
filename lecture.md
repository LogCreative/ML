
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