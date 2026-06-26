import numpy as np
from scipy.stats import spearmanr
np.random.seed(0)

# ---- 1. fake-quant math (mirrors quantizers.py) ----
def fq_asym(x, lo, hi, bit):
    levels = 2.0**bit - 1
    s = (hi - lo)/levels
    xc = np.clip(x, lo, hi)
    return np.round((xc-lo)/s)*s + lo
def fq_sym(w, u, bit):
    levels = 2.0**bit - 1
    s = 2*u/levels
    wc = np.clip(w, -u, u)
    return np.round((wc+u)/s)*s - u

x = np.random.randn(4000).astype(np.float64)
lo, hi = x.min(), x.max()
err = {b: np.mean((x-fq_asym(x,lo,hi,b))**2) for b in (4,6,8)}
assert err[8] < err[6] < err[4], err
print("1 fake-quant: error decreases with bits", {k:round(v,5) for k,v in err.items()}, "OK")

# ---- 2. bit-aware clipping (OMSE eps search, mirrors calibrate_bac) ----
heavy = np.concatenate([np.random.randn(3800), np.random.randn(200)*8])  # outliers
lo,hi = heavy.min(), heavy.max()
def bac_eps(x,lo,hi,bit,steps=100):
    best,best_e=1e30,1.0
    for i in range(steps):
        e=1.0-i*(0.9/steps)
        er=np.mean((x-fq_asym(x,lo*e,hi*e,bit))**2)
        if er<best: best,best_e=er,e
    return best_e,best
for b in (4,8):
    e,best=bac_eps(heavy,lo,hi,b)
    base=np.mean((heavy-fq_asym(heavy,lo,hi,b))**2)
    assert best<=base+1e-12 and e<1.0, (b,e,best,base)
    print(f"2 BaC bit={b}: eps={e:.2f} reduces MSE {base:.4f}->{best:.4f}  OK")

# ---- 3. complexity features (mirrors complexity.py) ----
def entropy(z,n=64):
    v=z.ravel(); v=(v-v.min())/(v.max()-v.min()+1e-8)
    h,_=np.histogram(v,bins=n,range=(0,1)); p=h/(h.sum()+1e-8); p=p[p>0]
    return -(p*np.log(p)).sum()
def hf(z,cut=0.25):
    H,W=z.shape[-2:]
    fy=np.abs(np.fft.fftfreq(H))[:,None]; fx=np.abs(np.fft.fftfreq(W))[None,:]
    mask=(np.sqrt(fy**2+fx**2)>cut*0.5)
    m=np.abs(np.fft.fft2(z))**2
    return (m*mask).sum()/(m.sum()+1e-8)
flat=np.ones((16,16))+0.01*np.random.randn(16,16)
noise=np.random.randn(16,16)
yy,xx=np.mgrid[0:16,0:16]; hi_freq=np.sin(xx*2.5)+np.sin(yy*2.5)
assert entropy(noise)>entropy(flat), (entropy(noise),entropy(flat))
assert hf(hi_freq)>hf(flat), (hf(hi_freq),hf(flat))
print(f"3 complexity: H(noise)={entropy(noise):.2f}>H(flat)={entropy(flat):.2f}; "
      f"hf(grid)={hf(hi_freq):.3f}>hf(flat)={hf(flat):.3f}  OK")

# ---- 4. greedy bit allocation under budget (mirrors ilp_init.py) ----
def allocate(sens,cost,legal,target):
    legal=sorted(legal); lo=legal[0]
    bits={k:lo for k in sens}
    total=sum(cost.values()); budget=target*total
    spent=lambda: sum(cost[k]*bits[k] for k in bits)
    order=sorted(sens,key=lambda k:sens[k]/(cost[k]+1e-12),reverse=True)
    for k in order:
        for b in legal[1:]:
            if spent()+cost[k]*(b-bits[k])<=budget: bits[k]=b
            else: break
    return bits
sens={f"b{i}":float(i) for i in range(8)}          # b7 most sensitive
cost={k:1.0 for k in sens}
bits=allocate(sens,cost,(4,8),5.0)
avg=sum(cost[k]*bits[k] for k in bits)/sum(cost.values())
assert avg<=5.0+1e-9, avg
assert bits["b7"]==8 and bits["b0"]==4, bits
print(f"4 allocate: avg_bit={avg:.2f}<=budget; most-sensitive=8, least=4  OK")

# ---- 5. budgeted policy: monotone in complexity, budget-centered (mirrors policy.py) ----
order=sorted(sens,key=lambda k:sens[k],reverse=True)
total=sum(cost.values())
def n8_for(target):
    budget=target*total; spent=4*total; n=0
    for k in order:
        if spent+cost[k]*4<=budget: spent+=cost[k]*4; n+=1
        else: break
    return n
base=n8_for(5.0); swing=max(1,int(round(0.15*len(order))))
n8min,n8max=max(0,base-swing),min(len(order),base+swing)
def assign(p): 
    n8=int(round(n8min+p*(n8max-n8min)))
    return n8
ps=np.linspace(0,1,11)
n8s=[assign(p) for p in ps]
assert all(n8s[i]<=n8s[i+1] for i in range(len(n8s)-1)), n8s   # monotone
avg_over_set=np.mean([ (sum( (8 if k in order[:assign(p)] else 4) for k in order))/len(order) for p in ps])
assert abs(avg_over_set-5.0)<1.0, avg_over_set
print(f"5 policy: n8 monotone {n8s}; mean avg_bit over complexities={avg_over_set:.2f}~budget  OK")

# ---- 6. gate analysis: cov, pairwise cosine, spearman (mirrors sensitivity.analyze) ----
nimg,nblk=24,8
# construct: strong layer structure (blocks differ a lot), modest image variance correlated w/ complexity
layer_profile=np.array([1,2,1,5,2,1,3,1.0])
comp=np.linspace(0,1,nimg)+0.05*np.random.randn(nimg)
M=np.outer(1+0.4*comp, layer_profile)*(1+0.05*np.random.randn(nimg,nblk))
def mpc(rows):
    R=rows/(np.linalg.norm(rows,axis=1,keepdims=True)+1e-12); S=R@R.T
    iu=np.triu_indices(len(R),1); return S[iu].mean()
img_mean=M.mean(1); cov=img_mean.std()/img_mean.mean()
li=mpc(M); rho=spearmanr(comp,img_mean).correlation
assert li>0.9, li                         # layer order invariant across images
assert rho>0.9, rho                       # complexity predicts sensitivity here
print(f"6 gate: CoV={cov:.3f}, layer_invariance={li:.3f}>0.9, spearman={rho:.3f}>0.9  OK")

print("\nALL ALGORITHM CHECKS PASSED")
