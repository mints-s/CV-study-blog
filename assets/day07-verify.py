"""Day 7의 합성 수치 예제를 검증한다. OpenCV 실행 결과는 아니다."""
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
P1 = K @ np.column_stack((np.eye(3), np.zeros(3)))
P2 = K @ np.column_stack((np.eye(3), [-.2, 0, 0]))

def dlt(p1, p2):
    A = np.vstack([p1[0]*P1[2]-P1[0], p1[1]*P1[2]-P1[1],
                   p2[0]*P2[2]-P2[0], p2[1]*P2[2]-P2[1]])
    h = np.linalg.svd(A)[2][-1]
    assert abs(h[3]) > 1e-12
    return h[:3]/h[3]

for z in (2., 4.):
    X = np.array([.25*z, .125*z, z])
    p1 = np.array([520., 340.])
    u2 = 520-160/z
    exact = dlt(p1, [u2, 340])
    perturbed = dlt(p1, [u2+1, 340])
    np.testing.assert_allclose(exact, X, atol=1e-10)
    np.testing.assert_allclose(perturbed[2], 160/(160/z-1), atol=1e-10)
    print(f'triangulation Z={z:g}: exact={exact}, right u+1 depth={perturbed[2]:.9f}')

# 8개 비공면 점의 무잡음 관측에서 자세를 최적화한다.
Xw = np.array([[x,y,z] for z in (0., .5) for y in (-.3,.3) for x in (-.4,.4)])
truth = np.array([0., 0., np.pi/6, .1, -.05, 3.])
initial = np.array([.05, -.04, .45, 0., 0., 2.8])

def project(theta):
    R = Rotation.from_rotvec(theta[:3]).as_matrix()
    Xc = Xw @ R.T + theta[3:]
    q = Xc @ K.T
    return q[:,:2]/q[:,2:3]

observed = project(truth)
def residual(theta):
    return (project(theta)-observed).ravel()
def rms(theta):
    return np.sqrt(np.mean(np.sum(residual(theta).reshape(-1,2)**2, axis=1)))

fit = least_squares(residual, initial, method='lm', jac='3-point',
                    x_scale='jac', loss='linear', max_nfev=200,
                    ftol=1e-12, xtol=1e-12, gtol=1e-12)
assert fit.success
Rest = Rotation.from_rotvec(fit.x[:3]).as_matrix()
Rtrue = Rotation.from_rotvec(truth[:3]).as_matrix()
angle = np.degrees(Rotation.from_matrix(Rest @ Rtrue.T).magnitude())
t_error = np.linalg.norm(fit.x[3:]-truth[3:])
center = -Rest.T @ fit.x[3:]
assert angle < 1e-6 and t_error < 1e-8 and rms(fit.x) < 1e-8
assert np.min((Xw @ Rest.T + fit.x[3:])[:,2]) > 0
print('NumPy', np.__version__, 'SciPy', scipy.__version__)
print('PnP local optimization: initial RMS=', rms(initial), 'final RMS=', rms(fit.x))
print('rvec=', fit.x[:3], 't=', fit.x[3:])
print('rotation error degrees=', angle, 'translation error m=', t_error)
print('camera center=', center)
print('first observation=', observed[0])
