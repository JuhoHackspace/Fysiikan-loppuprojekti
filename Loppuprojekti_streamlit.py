import pandas as pd
import numpy as np
import folium as fo
import streamlit as st
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Title of the app

st.title('Matka päiväkodilta kaupalle')

# Load the gps data

gps_data = pd.read_csv('./kiihtyvyys_sijainti/Location.csv')

# Drawing a map of the route

# Create a map centered at the mean latitude and longitude of the data

map = fo.Map(location=[gps_data['Latitude (°)'].mean(), gps_data['Longitude (°)'].mean()], zoom_start=14)

fo.PolyLine (gps_data[['Latitude (°)', 'Longitude (°)']], color='blue', weight=3, opacity=1).add_to(map)

# Add markers for the start and end points

fo.Marker([gps_data['Latitude (°)'][0], gps_data['Longitude (°)'][0]], popup='Start', icon=fo.Icon(color='green')).add_to(map)
fo.Marker([gps_data['Latitude (°)'][len(gps_data['Latitude (°)'])-1], gps_data['Longitude (°)'][len(gps_data['Longitude (°)'])-1]], popup='End', icon=fo.Icon(color='red')).add_to(map)

# Display the map with Streamlit

st_map = st_folium(map, width=900, height=650)

# Calculate the distance between each pair of points

def Haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371e3
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2) * math.sin(delta_phi/2) + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2) * math.sin(delta_lambda/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

gps_data['dist'] = np.zeros(len(gps_data))

for i in range(len(gps_data)-1):
    gps_data.loc[i,'dist'] = Haversine(gps_data['Latitude (°)'][i], gps_data['Longitude (°)'][i], gps_data['Latitude (°)'][i+1], gps_data['Longitude (°)'][i+1])
    gps_data['total_dist'] = np.cumsum(gps_data['dist'])

# Display the total distance travelled and mean speed

total_time = gps_data['Time (s)'].iloc[-1]
total_time_formatted = float("{:.2f}".format(total_time))
total_dist = gps_data['total_dist'].iloc[-1]
total_dist_formatted = float("{:.2f}".format(total_dist))
mean_speed = total_dist / total_time
mean_speed_formatted = float("{:.2f}".format(mean_speed))


st.write("Matkaan käytetty kokonaisaika: ", total_time_formatted, "s")

st.write("Kuljettu kokonaismatka: ", total_dist_formatted, "m")

st.write("Keskinopeus: ", mean_speed_formatted, "m/s")

# Load the accelerometer data

acceleration_data = pd.read_csv('./kiihtyvyys_sijainti/Linear Acceleration.csv')

# Subheader for the accelerometer data
st.subheader('Kiihtyvyysdata')

# Display the accelerometer z-component data

st.line_chart(acceleration_data, x='Time (s)', y='Linear Acceleration z (m/s^2)', x_label='Time (s)', y_label='Acceleration z (m/s^2)')

st.write("Näytetään vain 40 sekunnin ajanjakso kiihtyvyyskomponentista z, koska dataa on paljon ja sen visualisointi ei ole kuvaavaa.")
# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(acceleration_data['Time (s)'], acceleration_data['Linear Acceleration z (m/s^2)'])

# Set the x-axis limit
ax.set_xlim([20, 60])

# Set labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration z (m/s^2)')

# Display the plot in Streamlit
st.pyplot(fig)

# Filter the acceleration data

# Define a filter function

def butter_lowpass(data, cutoff, fs, nyq, order=5):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters

T = acceleration_data['Time (s)'][len(acceleration_data['Time (s)'])-1] - acceleration_data['Time (s)'][0]  # length of data in seconds
print('Length of data: ', T, 's')

N = len(acceleration_data['Time (s)'])  # total number of data points
print('Number of data points: ', N)

fs = N / T  # sampling frequency
print('Sampling frequency: ', fs, 'Hz')

nyq = fs/2  # Nyquist frequency
order = 3  # order of the filter
cutoff_low = 1/(0.2)  # lowpass cutoff frequency in Hz

acceleration_data['low_filtered_a_z'] = butter_lowpass(acceleration_data['Linear Acceleration z (m/s^2)'], cutoff_low, fs, nyq, order)

# Display the filtered data

st.write("Alipäästösuodatetun kiihtyvyyskomponentin z kuvaaja 40 sekunnin ajanjaksolta.")

fig, ax = plt.subplots()

# Plot the data
ax.plot(acceleration_data['Time (s)'], acceleration_data['low_filtered_a_z'])

# Set the x-axis limit
ax.set_xlim([20, 60])

# Set labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration z (m/s^2)')

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate the step count using the filtered data

# Step counter
# This function counts the number of steps taken based on the acceleration data and a threshold value.
# It considers a step to be taken when the acceleration value goes from below the threshold to above the threshold or vice versa.

def step_counter(acceleration_data, threshold):
  step_count = 0
  previous_value = acceleration_data[0]
  
  for value in acceleration_data:
     if (value > threshold and previous_value < threshold) or (value < -threshold and previous_value > -threshold):
        step_count += 1
        previous_value = value

  return step_count / 2

# Calculate the number of steps

threshold_ = 1 # A threshold value of 1 (m/s^2) derived from the filtered acceleration data plot

step_count = step_counter(acceleration_data['low_filtered_a_z'], threshold_)
step_count_formatted = float("{:.2f}".format(step_count))

st.write("Askelten määrä laskettuna suodatetusta datasta raja-arvon ylitysten ja alitusten perusteella: ", step_count_formatted)

st.subheader('Askel laskuri käyttäen fourier muunnosta')

# Using fourier transform to find the frequency of the steps

f = acceleration_data['Linear Acceleration z (m/s^2)']

fourier = np.fft.fft(f, N)

# Power spectral density

psd = fourier * np.conj(fourier) / N

# Frequency axis

freq = np.fft.fftfreq(N, T/N)

L = np.arange(1, np.floor(N/2), dtype='int')
PSD = np.array([freq[L], psd[L].real])

# Plotting the Power Spectral Density
st.write("Kiihtyvyysdatan tehospektritiheys Fourier muunnoksen avulla")

fig, ax = plt.subplots()

ax.plot(PSD[0], PSD[1], color='r', label='PSD')

ax.set_xlim(1, 3)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')

st.pyplot(fig)

st.write("Kuvaajan perusteella löydetään kaksi lähes yhtä vahvaa taajuutta")

# Find the peak frequency

dominant_freq = PSD[0, np.argmax(PSD[1,:])]
second_dominant_freq = PSD[0, np.argsort(PSD[1,:])[-2]]
dominant_freq_formatted = float("{:.2f}".format(dominant_freq))
second_dominant_freq_formatted = float("{:.2f}".format(second_dominant_freq))

st.write("Vahvin taajuus: ", dominant_freq_formatted, "Hz")
st.write("Toiseksi vahvin taajuus: ", second_dominant_freq_formatted, "Hz")

# Calculate the number of steps

step_count_dom_fourier = dominant_freq * T
step_count_sec_dom_fourier = second_dominant_freq * T
step_dom_formatted = float("{:.2f}".format(step_count_dom_fourier))
step_sec_dom_formatted = float("{:.2f}".format(step_count_sec_dom_fourier))

step_count_dom_2 = dominant_freq * (T - 7)
step_count_dom_2_formatted = float("{:.2f}".format(step_count_dom_2))

st.write("Askelten määrä laskettuna Fourier muunnoksen avulla vahvimman taajuuden perusteella: ", step_dom_formatted)
st.write("Askelten määrä laskettuna Fourier muunnoksen avulla toiseksi vahvimman taajuuden perusteella: ", step_sec_dom_formatted)

# Display the results

st.write("Toiseksi vahvimman taajuuden perusteella laskettu askelten määrä on lähimpänä suodatetusta datasta laskettua askelten määrää.")
st.write("Valitaan vertailukohdaksi luonnollisesti vahvin taajuus ja todetaan, että Fourier muunnoksen avulla laskettu askelten määrä on lähellä suodatetusta datasta laskettua askelten määrää.")
st.write("Ero askelten määrissä: ", step_count_dom_fourier, "-", step_count_formatted, "=" ,abs(step_count_dom_fourier - step_count))
st.write("Ero johtuu siitä, että Fourier muunnos ei ole täydellinen menetelmä askelten laskemiseen, mutta se antaa hyvän arvion askelten määrästä.")
st.write("Poistetaan ajasta vielä 7 sekuntia, jotta tehokkaimman taajuuden askelarvio olisi todenmukaisempi. Seisoskelin paikallaan ensimmäiset 7 sekuntia ennen kävelyn aloittamista. Tämä ei vaikuta suodatetun datan askelten määrään.")

st.write("Askelten määrä laskettuna Fourier muunnoksen avulla vahvimman taajuuden perusteella ja 7 sekunnin poistolla: ", step_count_dom_2_formatted)

st.subheader('Askelpituus kuljetun matkan ja suodatetusta datasta laskettujen askelten määrän perusteella')

st.write("Kuljettu matka: ", total_dist_formatted, "m")
st.write("Askelten määrä: ", step_count)

# Calculate the step length

step_length = total_dist / step_count
step_length_formatted = float("{:.2f}".format(step_length))

st.write("Askelten pituus: ", step_length_formatted, "m")

