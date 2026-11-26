"""고정 벡터의 NT-Xent 검산이다. 학습 성능 실험이 아니다."""
import math
U=[(1.,0.),(.8,.6),(0.,1.),(-.6,.8)]
pos=[1,0,3,2]
def row(scores,target,tau):
    logits=[s/tau for s in scores]; peak=max(logits)
    lse=peak+math.log(sum(math.exp(v-peak) for v in logits))
    prob=math.exp(logits[target]-lse)
    return lse-logits[target],prob

def evaluate(u,tau):
    out=[]
    for i,a in enumerate(u):
        ids=[k for k in range(len(u)) if k!=i]
        s=[sum(x*y for x,y in zip(a,u[k])) for k in ids]
        out.append(row(s,ids.index(pos[i]),tau)[0])
    return out,sum(out)/len(out)
assert all(math.isclose(sum(x*x for x in u),1) for u in U)
for tau in [1.,.5,.1]:print('tau',tau,'losses/mean',evaluate(U,tau),'anchor probability',math.exp(-evaluate(U,tau)[0][0]))
loss,p=row([.8,0.,-.6],0,.5)
eps=1e-6
numeric=(row([.8+eps,0.,-.6],0,.5)[0]-row([.8-eps,0.,-.6],0,.5)[0])/(2*eps)
analytic=(p-1)/.5
assert math.isclose(numeric,analytic,rel_tol=1e-7)
print('gradient',analytic,'finite difference',numeric)
for tau in [1.,.5,.1]:
    losses,mean=evaluate([(1.,0.)]*4,tau)
    assert math.isclose(mean,math.log(3))
print('collapsed',math.log(3))
print('wrong self inclusion',row([1.,.8,0.,-.6],1,.5)[0])
print('hard negative',row([.8,.9,-.6],0,.1))
