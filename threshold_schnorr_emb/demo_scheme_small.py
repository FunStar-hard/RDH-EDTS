from src.scheme.setup import setup
from src.scheme.sign_emb import sign_emb
from src.scheme.verify import verify
from src.scheme.extract import extract
import secrets

# 小样本参数
n = 5
t = 3
L = 2
Nmax = 256

print("=== SDIH-TSS small demo ===")
print(f"Parameters: n={n}, t={t}, L={L}, Nmax={Nmax}")

# 1. Setup
sr = setup(t, n, L)
print("[1] Setup finished")
print("    public key generated")
print("    shares generated:", len(sr.shares))
print("    extraction key generated:", sr.Kext is not None)

# 2. Prepare message and hidden value
m = b"demo message"
M = secrets.randbelow(2 ** L)
participants = list(range(1, t + 1))

print("[2] Message and embedded value")
print("    message =", m)
print("    hidden M =", M)
print("    participants =", participants)

# 3. SignEmb
result = sign_emb(
    m=m,
    participants=participants,
    shares=sr.shares,
    share_pks=sr.share_pks,
    pk=sr.pk,
    Kext=sr.Kext,
    M=M,
    L=L,
    Nmax=Nmax,
)

print("[3] SignEmb finished")
print("    success =", result.success)
print("    retries =", result.retries)

if not result.success:
    print("Signing failed. Try increasing Nmax.")
    raise SystemExit(1)

sig = result.signature
print("    signature R exists =", sig.R is not None)
print("    signature s =", sig.s)
print("    low bits s mod 2^L =", sig.s % (2 ** L))

# 4. Verify
v = verify(m, sr.pk, sig)
print("[4] Verify result =", v)

# 5. Extract
recovered = extract(m, sr.pk, sig, sr.Kext, L)
print("[5] Extract result =", recovered)
print("    original M  =", M)
print("    recovered M =", recovered)

print("=== Final result ===")
print("verify_ok =", v)
print("extract_ok =", recovered == M)