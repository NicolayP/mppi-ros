import numpy as np
import matplotlib.pyplot as plt

def tang_vec(self, points, a, b):
    vec = np.zeros(shape=(points.shape[0], 2))
    x = points[:, 0]
    y = points[:, 1]
    vec[:, 0] = -x*(b**2)
    vec[:, 1] = y*(a**2)
    return vec

point = np.array([0., 0., 3.])
normal = np.array([2, 0.5, 3])
xpf = np.array([-2, 2, 1])

normal = normal / np.linalg.norm(normal)
xpfNorm = xpf / np.linalg.norm(xpf)
ypfNorm = np.cross(normal, xpfNorm)

elOrigin = point

d = -point.dot(normal)

s = np.linspace(-5, 5, 100)
xx, yy = np.meshgrid(s, s)

zz = (-normal[0]*xx - normal[1]*yy - d) * 1./normal[2]

a = 1.
b = 2.

x0 = np.linspace(start=0., stop=1., num=100)
x1 = -x0

x = np.concatenate([x1, x1, x0, x0])

y0 = np.sqrt(b**2 * (1 - np.power(x0, 2)/a**2))
y1 = -y0


y = np.concatenate([y1, y0, y1, y0])

z = np.zeros(shape=(x.shape))

X = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])
t = np.expand_dims(elOrigin, axis=-1)
tmp0 = np.expand_dims(xpfNorm, axis=-1)
tmp1 = np.expand_dims(ypfNorm, axis=-1)
tmp2 = np.expand_dims(normal, axis=-1)
N = np.hstack([tmp0, tmp1, tmp2])

invN = np.linalg.inv(N)

R = (X) @ invN

xExt = np.expand_dims(x, axis=0)
yExt = np.expand_dims(y, axis=0)
zExt = np.expand_dims(z, axis=0)

samples = np.concatenate([xExt, yExt, zExt])

rotSamples = R.T @ samples + t

rotOrigin = R @ N


plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, zz, alpha=0.2)

ax = plt.gca()
ax.scatter3D(rotSamples[0, :], rotSamples[1, :], rotSamples[2, :])

ax.quiver(point[0], point[1], point[2], xpfNorm[0], xpfNorm[1], xpfNorm[2], length=1, normalize=True, color="green")
ax.quiver(point[0], point[1], point[2], ypfNorm[0], ypfNorm[1], ypfNorm[2], length=1, normalize=True, color="red")
ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=1, normalize=True, color="blue")

ax.quiver(point[0], point[1], point[2], rotOrigin[0, 0], rotOrigin[1, 0], rotOrigin[2, 0], length=1, normalize=True, color="green")
ax.quiver(point[0], point[1], point[2], rotOrigin[0, 1], rotOrigin[1, 1], rotOrigin[2, 1], length=1, normalize=True, color="red")
ax.quiver(point[0], point[1], point[2], rotOrigin[0, 2], rotOrigin[1, 2], rotOrigin[2, 2], length=1, normalize=True, color="blue")

ax.set_xlabel('X-axis', fontweight ='bold')
ax.set_ylabel('Y-axis', fontweight ='bold')
ax.set_zlabel('Z-axis', fontweight ='bold')


plt.show()

