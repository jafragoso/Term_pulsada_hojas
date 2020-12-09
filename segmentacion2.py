import skvideo.io
import skvideo.utils
import skvideo.datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fftpack import fft, fftfreq
from timeit import default_timer as timer
from skimage.io import imsave, imread
from mpl_toolkits.mplot3d import Axes3D


filename = skvideo.io.vread('/Users/Jus/Desktop/Test1.mov', outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
tiem, col, com = filename.shape
video_RDI = np.zeros((tiem, col, com))
curva = np.zeros((tiem))
curva2 = np.zeros((tiem))
x = np.linspace(0, tiem / 29, len(curva))
img_umbral = np.zeros((col, com))

for i in range(0, tiem):
    numero_pixeles = col * com
    salida_R = np.sum(filename[i, :, :], axis=1)
    salida2_R = np.sum(salida_R, axis=0)
    sal_promedio_R = salida2_R / numero_pixeles
    curva[i] = sal_promedio_R


curva2 = curva.tolist()
alto = max(curva2)
indice = curva2.index(alto)

umbral = 10
# poner indice en el 200
for i in range(0, col):
    for j in range(0, com):
        if filename[indice, i, j] <= umbral:
            img_umbral[i, j] = 0
        elif filename[indice, i, j] > umbral:
            img_umbral[i, j] = 1

numero_de_unos = np.count_nonzero(img_umbral)

for i in range(0, tiem):
    video_RDI[i, :, :] = filename[i, :, :]*img_umbral
    numero_pixeles = col * com
    salida_R = np.sum(video_RDI[i, :, :], axis=1)
    salida2_R = np.sum(salida_R, axis=0)
    #########################
    sal_promedio_R = salida2_R / numero_de_unos  # numero_pixeles
    curva2[i] = sal_promedio_R

w = savgol_filter(curva2, 101, 6)
w1 = w

for i in range(0, len(w1)):
    w1[i] = round(w[i], 6)

w2 = w1.tolist()
alto = max(w2)
bajo = min(w2)
indiceB = w2.index(bajo)
indice = w2.index(alto)
w3 = w2[indice:len(w2)]
x2 = np.linspace(0, len(w3)/29, len(w3))
# np.savetxt('/Users/Jus/Documents/Vdes/platano.csv', curva2)

Fs = 20  # frecuencia de muestreo
Ts = 1.0/Fs  # intervalo de muestreo dt
tiempo_total = tiem / Fs

n = len(w2)  # longitud de la señal
t = np.linspace(0, tiempo_total, n)  # time vector

fft = fft(w2) / n
frq = fftfreq(n, Ts)

# computar fft

fr = (Fs/2)*np.linspace(0, 1, n/2)
X = np.fft.fft(w2)
X_m = (2/n)*abs(X[0:np.size(fr)])  # obtengo magnitus
fft_imag = fft.imag
print('pico máximo de temperatura', max(w2))
print('frecuencia mas alta', max(fft_imag))
print('amplitud mas alta', max(X_m))

imagen_prueba = filename[indice, :, :]
fila_prueba = imagen_prueba[37]
fila_prueba2 = fila_prueba.tolist()
pixel_numero = np.linspace(0,80, len(fila_prueba2))

plt.figure(0)
#plt.subplot(3, 1, 1)
plt.plot(pixel_numero, fila_prueba2); plt.title('Horizontal intensity')
#plt.plot(t*21, curva2); plt.title('Señal Original')
plt.xlabel('Column No.'); plt.ylabel('Intensity')
# create the figure

xx, yy = np.mgrid[0:video_RDI[indice,:,:].shape[0], 0:video_RDI[indice,:,:].shape[1]]

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, filename[indice,:,:], rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)

fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, filename[300,:,:], rstride=1, cstride=1, cmap=plt.cm.inferno, linewidth=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes.ravel()

ax[0].set_title('1')
ax[0].imshow(filename[indice,:,:], cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('2')
ax[1].imshow(filename[600,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[1].set_axis_off()

ax[2].set_title('3')
ax[2].imshow(filename[400,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[2].set_axis_off()

ax[3].set_title('4')
ax[3].imshow(filename[500,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[3].set_axis_off()
plt.tight_layout()

plt.show()

print('inice frio', indiceB)
print('inice caliente', indice)
