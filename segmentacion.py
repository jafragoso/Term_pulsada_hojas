import skvideo.io
import math
import skvideo.utils
import skvideo.datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fftpack import fft, fftfreq
from timeit import default_timer as timer
start = timer()
direccion0 = "/Users/Jus/Desktop/pr_50_3/11/4"
direccion = direccion0 + "/video.avi"
direccion2 = direccion0 + "/video.csv"

filename = skvideo.io.vread(direccion, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
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
        if filename[200, i, j] <= umbral:
            img_umbral[i, j] = 0
        elif filename[200, i, j] > umbral:
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


for i in range(1):
    MR=491.39*math.exp(-2.577*(max(fft_imag))-0.2)
    #MR = (-91.46*math.log(max(fft_imag)))+43.838
    #MR = (81.629*max(fft_imag)**2)-(273.35*max(fft_imag))+233.38
    if MR<0:
        print('Fuera de rango')
    if 0<=MR<=10:
        print('Deshidratado muy alto')
    if 10<MR<=40:
        print('Deshidratado alto')
    if 40<MR<=70:
        print('Deshidratado medio')
    if 70<MR<=100:
        print('Deshidratado bajo')
    if MR>100:
        print('Deshidratado bajo')

#print('MR',MR)

plt.figure(1)
#plt.subplot(3, 1, 1)
plt.plot(t, w2); plt.title('Señal Original')
plt.xlabel('tiempo (s)'); plt.ylabel('amplitud')

plt.figure(2)
#plt.subplot(3, 1, 2)
plt.plot(fr,X_m); plt.title('espectro de magnitud')
plt.xlabel('frecuencia'); plt.ylabel('magnitud ')
#plt.tight_layout()

plt.figure(3)
#plt.subplot(3, 1, 3)
plt.vlines(frq, 0,fft_imag); plt.title('espectro de frecuencia')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Im($Y$)')
#plt.tight_layout()

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax = axes.ravel()

ax[0].set_title('1')
ax[0].imshow(img_umbral, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('1')
ax[1].imshow(video_RDI[40,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[1].set_axis_off()

ax[2].set_title('1')
ax[2].imshow(video_RDI[300,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[2].set_axis_off()

ax[3].set_title('1')
ax[3].imshow(video_RDI[600,:,:], cmap=plt.cm.inferno, interpolation='nearest')
ax[3].set_axis_off()
plt.tight_layout()
plt.show()

np.savetxt(direccion2, w2)
np.savetxt('/Users/Jus/Documents/Vdes/amplitud.csv', X_m)
np.savetxt('/Users/Jus/Documents/Vdes/fase.csv', fft_imag)

vectoradd_time = timer() - start
#print("Proc time %f seconds" % vectoradd_time)
