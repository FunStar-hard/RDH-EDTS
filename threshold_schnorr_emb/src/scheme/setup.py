"""System setup: key generation, Shamir sharing, Kext generation."""
from __future__ import annotations

import secrets

from src.crypto.curve_utils import get_generator, get_order, scalar_mult
from src.crypto.shamir import share_secret
from src.scheme.types import SetupResult

# 输入门限参数 (𝑡,𝑛)、嵌入比特 𝐿
def setup(t: int, n: int, L: int) -> SetupResult:
    """Run the Setup algorithm.

    Parameters
    ----------
    t : int – threshold
    n : int – total number of nodes
    L : int – embedding bit‑length

    Returns
    -------
    SetupResult
    """
    # 选取 𝑞,𝐺。get_order() → 调用src/crypto/curve_utils.py；get_generator() → 调用src/crypto/curve_utils.py
    q = get_order()# 获取椭圆曲线群的阶q（公式中Z_q的模）
    G = get_generator()# 获取椭圆曲线的生成元G

    #生成系统密钥对 (x, pk)，其中 x 是私钥，pk = xG 是对应的公钥。scalar_mult(x, G) → 调用src/crypto/curve_utils.py
    x = secrets.randbelow(q - 1) + 1 # 生成系统私钥sk（代码中变量名为x）
    pk = scalar_mult(x, G) # 计算系统公钥pk = x·G（对应公式3）
    #调用 Shamir 分发
    # 2. Shamir shares f(x) = sk + a_1 x + ... + a_{t-1} x^{t-1} mod q
    _coeffs, shares = share_secret(x, t, n, q)  # 调用Shamir秘密共享函数，src/crypto/shamir.pygenerate_polynomial函数（shamir.py第 9-15 行））
    #计算份额公钥pk_i = x_i * G
    # 3. Share public keys。遍历所有节点的私钥份额xi（即sk_i）；对每个份额调用scalar_mult(xi, G)计算对应的公钥点；返回字典{节点ID: 份额公钥pk_i}（对应论文的{(x_i, pk_i)}_{i=1}^n）
    share_pks = {vi: scalar_mult(xi, G) for vi, xi in shares.items()}
    #生成提取密钥
    # 4. Extraction key# 生成256位（32字节）的提取密钥k_ext
    Kext = secrets.token_bytes(32)  # 256 bits

    return SetupResult(
        pk=pk,
        shares=shares,
        share_pks=share_pks,
        Kext=Kext,
        n=n,
        t=t,
        L=L,
        system_secret=x,
    )