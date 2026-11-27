"""유한 구간 emission-absorption 검산이다. 학습 실험이 아니다."""
import math

def render(sigmas,deltas,colors,bg=(0.,0.,0.)):
    T=1.;out=[0.,0.,0.];weights=[];alphas=[];trans=[]
    for s,d,c in zip(sigmas,deltas,colors):
        assert s>=0 and d>=0
        a=-math.expm1(-s*d);w=T*a
        trans.append(T);alphas.append(a);weights.append(w)
        for k in range(3):out[k]+=w*c[k]
        T*=math.exp(-s*d)
    for k in range(3):out[k]+=T*bg[k]
    assert math.isclose(sum(weights)+T,1.,abs_tol=1e-12)
    return out,weights,T,alphas,trans
sig=[math.log(2),math.log(2),math.log(2)];delta=[1.,2.,1.]
colors=[(1,0,0),(0,1,0),(0,0,1)]
r=render(sig,delta,colors);print('black',r)
white=render(sig,delta,colors,(1,1,1));print('white',white[0])
assert r[1]==[.5,.375,.0625] and r[2]==.0625
assert white[0]==[.5625,.4375,.125]
print('reverse',render(sig,delta,list(reversed(colors)))[0])
base=render([.7],[2.],[(.2,.4,.8)])
for n in [2,8,64]:
    split=render([.7]*n,[2./n]*n,[(.2,.4,.8)]*n)
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(base[0],split[0]))
print('constant segment splitting passed')
# 방향 길이 2와 파라미터 폭 0.5를 곱하면 거리 1이다.
a=render([math.log(2)],[1.],[(1,0,0)])[0][0]
b=render([math.log(2)],[.5],[(1,0,0)])[0][0]
print('unit correction alpha',a,'incorrect alpha',b)
assert math.isclose(b,1-1/math.sqrt(2))
# 밀도와 길이를 반대 비율로 바꾸면 색을 보존한다.
scaled=render([s/3 for s in sig],[d*3 for d in delta],colors)
assert all(math.isclose(a,b) for a,b in zip(r[0],scaled[0]))
# first density derivative: delta1*(1-alpha1)*(c1-behind)
behind=render(sig[1:],delta[1:],colors[1:])[0]
grad=[.5*(colors[0][k]-behind[k]) for k in range(3)]
eps=1e-6
plus=render([sig[0]+eps]+sig[1:],delta,colors)[0]
minus=render([sig[0]-eps]+sig[1:],delta,colors)[0]
fd=[(a-b)/(2*eps) for a,b in zip(plus,minus)]
assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(grad,fd))
print('density gradient',grad,'finite difference',fd)
