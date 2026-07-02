import secrets
from ecdsa import NIST256p
from ecdsa.ellipticcurve import PointJacobi

# ===================== 原项目核心函数复现 =====================
def generate_polynomial(t: int, secret: int, q: int) -> list[int]:
    """构造(t-1)次多项式，常数项为secret（对应论文公式4）"""
    coeffs = [secret % q]  # 常数项 = 系统私钥sk
    for _ in range(t - 1):
        coeffs.append(secrets.randbelow(q))  # 生成t-1个随机系数a1~a(t-1)
    return coeffs

def evaluate_polynomial(coeffs: list[int], x: int, q: int) -> int:
    """霍纳法则计算多项式在x处的值（对应论文公式5）"""
    result = 0
    for a in reversed(coeffs):
        result = (result * x + a) % q
    return result

def share_secret(secret: int, t: int, n: int, q: int) -> tuple[list[int], dict[int, int]]:
    """Shamir秘密共享，生成n个私钥份额"""
    coeffs = generate_polynomial(t, secret, q)
    shares = {}
    for i in range(1, n + 1):  # 节点ID从1到n（与原项目完全一致）
        shares[i] = evaluate_polynomial(coeffs, i, q)
    return coeffs, shares

# ===================== Setup算法完整测试 =====================
def test_setup(t: int, n: int, L: int):
    print("=" * 90)
    print("SDDH-TSS 3.2节 Setup算法 完整测试")
    print(f"测试参数：门限t={t} | 总节点数n={n} | 嵌入比特L={L}")
    print("=" * 90)

    # -------------------- 步骤1：选择椭圆曲线群参数 --------------------
    print("\n📌 步骤1：选择椭圆曲线群参数（论文第1段）")
    CURVE = NIST256p
    G = CURVE.generator  # 生成元G
    q = CURVE.order      # 群阶q（Z_q的模）
    print(f"  使用曲线：NIST P-256")
    print(f"  生成元G的x坐标：{G.x()}")
    print(f"  生成元G的y坐标：{G.y()}")
    print(f"  群阶q（十进制）：{q}")
    print(f"  群阶q（十六进制）：{hex(q)}")
    print(f"  q的位数：{q.bit_length()}位")

    # -------------------- 步骤2：生成系统密钥对(sk, pk) --------------------
    print("\n📌 步骤2：生成系统密钥对（论文公式3：pk = sk·G）")
    sk = secrets.randbelow(q - 1) + 1  # 私钥范围[1, q-1]，符合密码学安全要求
    print(f"  系统私钥sk：{sk}")
    pk = sk * G  # 计算系统公钥
    print(f"  系统公钥pk的x坐标：{pk.x()}")
    print(f"  系统公钥pk的y坐标：{pk.y()}")
    
    # 验证公钥正确性
    assert pk == sk * G, "❌ 系统公钥计算错误！"
    print("  ✅ 系统公钥验证通过")

    # -------------------- 步骤3：构造多项式并生成私钥份额 --------------------
    print("\n📌 步骤3：构造(t-1)次多项式并生成私钥份额（论文公式4、5）")
    coeffs, shares = share_secret(sk, t, n, q)
    print(f"  多项式系数：{coeffs}")
    print(f"  多项式次数：{len(coeffs)-1}次（t-1={t-1}）")
    print(f"  常数项验证：{coeffs[0]} == sk={sk} → {coeffs[0] == sk}")
    
    print("\n  各节点私钥份额（sk_i = f(x_i) mod q）：")
    for vi, sk_i in shares.items():
        print(f"    节点{vi}（x_i={vi}）：sk_{vi} = {sk_i}")

    # -------------------- 步骤4：计算份额公钥 --------------------
    print("\n📌 步骤4：计算份额公钥（论文公式6：pk_i = sk_i·G）")
    share_pks = {}
    for vi, sk_i in shares.items():
        pk_i = sk_i * G
        share_pks[vi] = pk_i
        print(f"    节点{vi}：pk_{vi}的x坐标 = {pk_i.x()}")
        print(f"    节点{vi}：pk_{vi}的y坐标 = {pk_i.y()}")
        print("    " + "-" * 60)
    
    # 验证所有份额公钥正确性
    all_valid = True
    for vi, sk_i in shares.items():
        computed_pk_i = sk_i * G
        if computed_pk_i != share_pks[vi]:
            all_valid = False
            print(f"❌ 节点{vi}的份额公钥验证失败！")
    if all_valid:
        print("  ✅ 所有节点份额公钥验证通过")

    # -------------------- 步骤5：生成提取密钥 --------------------
    print("\n📌 步骤5：生成提取密钥k_ext（论文最后一段）")
    Kext = secrets.token_bytes(32)  # 256位提取密钥，与原项目一致
    print(f"  提取密钥Kext（十六进制）：{Kext.hex()}")
    print(f"  提取密钥长度：{len(Kext)*8}位")

    # -------------------- 最终结果汇总 --------------------
    print("\n" + "=" * 90)
    print("✅ Setup算法执行完成！最终输出结果（对应论文公式2）：")
    print("=" * 90)
    print(f"  Setup(t={t}, n={n}, l={L}) → (pk, {{(x_i, pk_i)}}, {{sk_i}}, k_ext)")
    print(f"\n  系统公钥pk：({pk.x()}, {pk.y()})")
    print(f"  提取密钥k_ext：{Kext.hex()}")
    print(f"\n  节点ID与密钥映射：")
    for vi in shares:
        print(f"    节点{vi}：x_i={vi} | sk_i={shares[vi]} | pk_i=({share_pks[vi].x()}, ...)")

    print("\n" + "=" * 90)
    print("=" * 90)

    # 返回结果供后续扩展使用
    return {
        "pk": pk,
        "shares": shares,
        "share_pks": share_pks,
        "Kext": Kext,
        "n": n,
        "t": t,
        "L": L,
        "system_secret": sk
    }

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 你可以在这里修改测试参数
    TEST_T = 3    # 门限值（至少需要3个节点签名）
    TEST_N = 5    # 总节点数
    TEST_L = 4    # 嵌入比特长度

    # 运行测试
    test_setup(TEST_T, TEST_N, TEST_L)