import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fft import fft, ifft
import scipy.io.wavfile as wave

# Bolum 1


# sampling rate = 8000


def takebeeps(samples, sampling_rate):
    beeps = []
    length = int(len(samples) / 22)
    for i in range(0, 11):
        beep = samples[length * 2 * i : length * 2 * i + length]
        beeps.append(beep)
    return beeps


def findmaxfourierofbeeps(beeps, sampling_rate=8000):

    maxfourier = []
    for beep in beeps:
        fourier = fft(beep, sampling_rate)
        fourier = np.abs(fourier)
        zort = np.argsort(fourier[0:2000])
        maxfourier.append([zort[-1], zort[-2]])
    return maxfourier


def find_nearest(array, value):
    array = np.asarray(array)
    for el in array:
        if np.abs(el - value) < 50:
            idx = (np.abs(array - value)).argmin()
            return idx

    else:
        return -1


def fourieranalysis(fouriers, columns, rows, keymatrix):
    text = ""
    numberarr = []
    for fourier in fouriers:
        x = find_nearest(columns, fourier[0])
        y = find_nearest(rows, fourier[1])
        if x == -1 or y == -1:
            x = find_nearest(columns, fourier[1])
            y = find_nearest(rows, fourier[0])
        text = text + str(keymatrix[y][x])
        numberarr.append(keymatrix[y][x])
    print(text)
    return numberarr


def printnumbers(fileadress):
    columns = [1209, 1336, 1477]
    rows = [697, 770, 852, 941]
    keysmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], ["*", 0, "#"]]

    file_path = fileadress
    samples, sampling_rate = librosa.load(
        file_path,
        sr=None,
        mono=True,
        offset=0.0,
        duration=None,
    )

    beeps = takebeeps(samples, sampling_rate)
    fouriers = findmaxfourierofbeeps(beeps, sampling_rate)
    fourieranalysis(fouriers, columns, rows, keysmatrix)


def makenumbersoundarr(number):
    columns = [1209, 1336, 1477]
    rows = [697, 770, 852, 941]
    keysmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], ["*", 0, "#"]]
    keysmatrix = np.array(keysmatrix)
    x_location, y_location = (
        np.where(keysmatrix == number)[0][0],
        np.where(keysmatrix == number)[1][0],
    )
    frequencies = [rows[x_location], columns[y_location]]

    fourierarray = []

    for i in range(0, 2000):
        if i == frequencies[0] or i == frequencies[1]:
            fourierarray.append(100)
        else:
            fourierarray.append(0)

    voicearray = ifft(fourierarray, 8000).real

    return voicearray[0:500]


def writesoundfromtelephonenumber(telephonenumber, filename="default.wav"):
    longarr = []
    arr2 = 500 * [0]
    telephonenumber = list(telephonenumber)
    for number in telephonenumber:
        arr = makenumbersoundarr(number)
        longarr.extend(arr)
        longarr.extend(arr2)
    longarr = np.array(longarr)
    wave.write(filename, 8000, longarr)

    return longarr


def stemandplotofsound(fileadress):
    columns = [1209, 1336, 1477]
    rows = [697, 770, 852, 941]
    keysmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], ["*", 0, "#"]]

    file_path = fileadress
    samples, sampling_rate = librosa.load(
        file_path,
        sr=None,
        mono=True,
        offset=0.0,
        duration=None,
    )

    plt.stem(samples, markerfmt=" ")
    plt.show()
    plt.plot(samples)
    plt.show()


def stemofnumbers(fileadress):
    columns = [1209, 1336, 1477]
    rows = [697, 770, 852, 941]
    keysmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], ["*", 0, "#"]]

    file_path = fileadress
    samples, sampling_rate = librosa.load(
        file_path,
        sr=None,
        mono=True,
        offset=0.0,
        duration=None,
    )

    beeps = takebeeps(samples, sampling_rate)
    fouriers = findmaxfourierofbeeps(beeps, sampling_rate)
    numbers = fourieranalysis(fouriers, columns, rows, keysmatrix)

    fig, axs = plt.subplots(3, 4)
    for i in range(11):
        j = i / 3
        j = int(j)

        axs[i % 3, j].stem(fft(beeps[i], 8000)[0:2000], markerfmt=" ")
        axs[i % 3, j].set_title(f"{i+1}. Sayı = {numbers[i]}")
    plt.show()


# Kendi telefon numaramin ses dosyasini olusturuyorum
telephonenumber = "05394330186"
writesoundfromtelephonenumber(telephonenumber, "05394330186.wav")

# Ornek dosyanin ses sifresini çözüyorum
print("Ornek dosyanin ses sifresi: ")
printnumbers("Ornek.wav")

# Kendi dosyamin ses sifresini cozuyorum
print("Kendi dosyamin ses sifresi: ")
printnumbers("05394330186.wav")

stemandplotofsound("Ornek.wav")
stemandplotofsound("telefon.wav")

stemofnumbers("Ornek.wav")
stemofnumbers("05394330186.wav")
