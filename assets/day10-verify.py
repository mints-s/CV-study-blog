"""위치 부호와 작은 pre-LN 블록의 순열 성질을 검산한다."""
import numpy as np
np.set_printoptions(precision=6,suppress=True)
X=np.array([[1.,0.,0.,1.],[0.,1.,1.,0.],[1.,1.,0.,0.]])
perm=np.array([2,0,1])
def pe(n,d):
 p=np.arange(n)[:,None];f=10000.**(-np.arange(0,d,2)/d)
 e=np.zeros((n,d));e[:,0::2]=np.sin(p*f);e[:,1::2]=np.cos(p*f)
 return e
E=pe(3,4)
def ln(x):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
def block(x):
 z=ln(x)
 # 단일 head, Q/K/V/O 투영은 항등이다.
 s=z@z.T/2
 a=np.exp(s-s.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
 u=x+a@z
 # d_ff=4, W1=W2=I, bias=0, ReLU인 검산용 FFN이다.
 return u+np.maximum(ln(u),0)
y=block(X)
base=block(X+E)
err0=np.max(np.abs(block(X[perm])-y[perm]))
err_joint=np.max(np.abs(block((X+E)[perm])-base[perm]))
err_fixed=np.max(np.abs(block(X[perm]+E)-base[perm]))
print('numpy',np.__version__)
print('PE',E,sep='\n')
print('no position error',err0,'joint permutation error',err_joint,'fixed slots error',err_fixed)
print('first output',base[0])
assert err0<1e-12 and err_joint<1e-12 and err_fixed>.1
np.testing.assert_allclose(E[0],[0,1,0,1])
f=10000.**(-np.arange(0,4,2)/4)
# 각 sin/cos 쌍의 내적은 상대 위치 차이의 cosine 합이다.
np.testing.assert_allclose(E[1]@E[2],np.cos(f).sum())
# 기본 LayerNorm의 평균과 분산을 검산한다.
v=np.array([[1.,2.,3.,4.]])
print('LN example',ln(v),'variance',ln(v).var())
assert abs(ln(v).mean())<1e-12
np.testing.assert_allclose(ln(v).var(),1.25/(1.25+1e-5))
print('PE dot at distance 1',E[1]@E[2])
