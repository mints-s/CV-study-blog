"""고정 QKV의 attention 계산과 마스크·순열 성질을 검산한다."""
import numpy as np
Q=np.array([[1.,0.],[0.,1.]])
K=np.array([[1.,0.],[0.,1.],[1.,1.]])
V=np.array([[10.,0.],[0.,10.],[4.,4.]])
def attention(q,k,v,mask=None,scale=True):
 s=q@k.T/(np.sqrt(q.shape[-1]) if scale else 1.)
 if mask is not None:s=np.where(mask,s,-np.inf)
 assert np.all(np.any(np.isfinite(s),axis=-1))
 w=np.exp(s-s.max(axis=-1,keepdims=True));w/=w.sum(axis=-1,keepdims=True)
 return w,w@v
A,O=attention(Q,K,V)
mask=np.array([[True,True,False],[True,True,False]])
Am,Om=attention(Q,K,V,mask)
np.testing.assert_allclose(A.sum(-1),1)
np.testing.assert_allclose(Am[:,2],0)
perm=[2,0,1]
np.testing.assert_allclose(attention(Q,K[perm],V[perm])[1],O)
# 위치와 마스크 없는 self-attention의 순열 등변성을 검사한다
X=K.copy();Wq=np.array([[1.,.2],[.3,1.]]);Wk=np.array([[.5,.1],[.2,1.]])
def selfatt(x):return attention(x@Wq,x@Wk,x)[1]
np.testing.assert_allclose(selfatt(X[perm]),selfatt(X)[perm])
# 학습 효과를 주장하지 않는 고정 입력의 스케일 비교다
print('NumPy',np.__version__)
print('scores',Q@K.T/np.sqrt(2))
print('weights',A,'output',O,sep='\n')
print('masked weights',Am,'masked output',Om,sep='\n')
print('unscaled',attention(Q,K,V,scale=False)[0])
print('scaled Q,K x4',attention(4*Q,4*K,V)[0])
assert np.isclose(A[0,0],A[0,2])
print('row sums, masking, KV permutation, self-attention equivariance passed')
