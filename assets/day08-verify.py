"""고정된 두 기준 카메라 아래에서 점 전용 보정과 공동 보정을 비교한다."""
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
rng=np.random.default_rng(8)
X=np.array([[x,y,z] for z in (3.,4.,5.) for y in (-.4,.4) for x in (-.5,.5)])
poses=np.array([[0,0,0,0,0,0],[0,0,0,-.5,0,0],[0,.08,0,-.2,-.3,.1]],float)
def project(points,ps):
 out=[]
 for p in ps:
  q=points@Rotation.from_rotvec(p[:3]).as_matrix().T+p[3:]
  out.append(800*q[:,:2]/q[:,2:]+[320,240])
 return np.array(out)
obs=project(X,poses)+rng.normal(0,.3,(3,12,2))
Xi=X+rng.normal(0,.04,X.shape)
pi=poses.copy();pi[2]+=[.02,-.015,.01,.05,-.04,.08]
def unpack(v):
 ps=poses.copy();ps[2]=v[:6]
 return v[6:].reshape(-1,3),ps
def residual(v):
 pts,ps=unpack(v);return (project(pts,ps)-obs).ravel()
v0=np.r_[pi[2],Xi.ravel()]
opt=dict(method='trf',jac='3-point',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=300)
point=least_squares(lambda a:(project(a.reshape(-1,3),pi)-obs).ravel(),Xi.ravel(),**opt)
joint=least_squares(residual,v0,**opt)
def metrics(pts,ps):
 rms=np.sqrt(np.mean(np.sum((project(pts,ps)-obs)**2,axis=-1)))
 err=np.sqrt(np.mean(np.sum((pts-X)**2,axis=-1)))
 return rms,err
for label,pts,ps in [('initial',Xi,pi),('points only',point.x.reshape(-1,3),pi),('joint',*unpack(joint.x))]:
 print(label,metrics(pts,ps))
assert point.success and joint.success
assert np.linalg.norm(residual(joint.x)) < np.linalg.norm(residual(v0))
# 공통 스케일 불변성을 검사한다
scaled=poses.copy();scaled[:,3:]*=2
assert np.allclose(project(2*X,scaled),project(X,poses))
pts,ps=unpack(joint.x)
assert all(np.all((pts@Rotation.from_rotvec(p[:3]).as_matrix().T+p[3:])[:,2]>0) for p in ps)
print('versions',np.__version__,scipy.__version__)
