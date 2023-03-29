#install requirements
get_ipython().system('pip install librosa')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install umap-learn')
get_ipython().system('pip install pandas')


Hermeneutics 2
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('Agg')
import requests
import io
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Audio

def plot_waveform_and_play_audio(mp3_url):
    # Download the MP3
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()

    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Plot the waveform using librosa.display
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Waveform of {}'.format(mp3_url))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('waveform.png', bbox_inches='tight')
    plt.close()
    display(Image(filename='waveform.png'))

    # Play the audio
    display(Audio(data=y, rate=sr))

plot_waveform_and_play_audio('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3')


Hermeneutics 3
def plot_power_spectrum(mp3_url, frame_length=2048, hop_length=512):
    # Download the MP3 file from the URL
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()

    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Get the first frame of the audio file
    first_frame = y[:frame_length]

    # Compute the power spectrum
    power_spectrum = np.abs(np.fft.fft(first_frame))**2

    # Plot the power spectrum
    plt.figure(figsize=(12, 4))
    freqs = np.fft.fftfreq(frame_length, 1 / sr)
    plt.plot(freqs[:frame_length // 2], power_spectrum[:frame_length // 2])
    plt.title('Power Spectrum of {}'.format(mp3_url))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.savefig('power_spectrum.png', bbox_inches='tight')
    plt.close()
    display(Image(filename='power_spectrum.png'))

plot_power_spectrum('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3')


Hermeneutics 4
get_ipython().run_line_magic('matplotlib', 'inline')


def plot_waveform_with_onsets(mp3_url):
    # Download the MP3 file from the URL
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()
    
    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Detect onsets
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    
    # Plot the waveform using librosa.display
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    
    # Plot red lines for detected onsets
    for onset in onsets:
        plt.axvline(x=onset, color='r', alpha=0.5)
        
    plt.title('Waveform of {} with Detected Onsets'.format(mp3_url))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('waveform_with_onsets.png', bbox_inches='tight')
    plt.close()
    display(Image(filename='waveform_with_onsets.png'))

plot_waveform_with_onsets('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3')



Hermeneutics 5
def visualize_mfcc(mp3_url):
    # Download the MP3 file from the URL
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()

    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalize MFCC values to have zero mean and unit variance
    mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)

    # Plot the normalized MFCC bands
    plt.figure(figsize=(30, 20))
    librosa.display.specshow(mfcc_norm, x_axis='time', y_axis='mel', sr=sr, cmap='coolwarm')

    plt.colorbar(format='%+0.2f')
    plt.title('Normalized MFCC Bands')
    plt.tight_layout()

    # Save the figure as an image file
    plt.savefig('normalized_mfcc_bands.png', bbox_inches='tight')
    plt.close()

    # Display the saved image
    display(Image(filename='normalized_mfcc_bands.png'))

visualize_mfcc('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3')


Hermeneutics 6
from IPython.display import Image
def analyze_audio(mp3_url):
    # Download the MP3 file from the URL
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()

    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Extract MFCC features and compute mean
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=0)

    # Normalize MFCC values to have zero mean and unit variance
    mfcc_mean_norm = (mfcc_mean - np.mean(mfcc_mean)) / np.std(mfcc_mean)

    # Compute the spectral centroid for each frame
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Plot the waveform, normalized MFCC mean, and spectral centroid
    fig, ax = plt.subplots(figsize=(30, 20))

    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.9)
    ax.set(title='Audio waveform')

    # Plot normalized MFCC mean on top of waveform
    time = np.arange(len(mfcc_mean)) * len(y) / len(mfcc_mean) / sr
    ax.plot(time, mfcc_mean_norm * (np.max(y) - np.min(y)) + np.mean(y), color='g', alpha=0.4, label='Normalized MFCC Mean')

    # Plot spectral centroid on top of waveform
    centroid_time = np.arange(spectral_centroid.shape[1]) * len(y) / spectral_centroid.shape[1] / sr
    ax.plot(centroid_time, spectral_centroid[0] / np.max(spectral_centroid) * (np.max(y) - np.min(y)) + np.mean(y), color='r', alpha=0.4, label='Spectral Centroid')

    # Add legend
    ax.legend()

    # Add time stamps at the bottom of the plot
    ax.set_xlabel('Time (s)')
    time_range = np.linspace(0, len(y)/sr, num=5)
    time_labels = [f'{t:.2f}' for t in time_range]
    plt.xticks(time_range, time_labels)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    # Save the figure as an image file
    plt.savefig('audio_analysis.png', bbox_inches='tight')
    plt.close()

    # Display the saved image
    display(Image(filename='audio_analysis.png'))

analyze_audio('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3')


Hermeneutics 7
def analyze_audio(mp3_url, threshold, similarity_threshold):
    # Download the MP3 file from the URL
    response = requests.get(mp3_url, allow_redirects=True)
    response.raise_for_status()

    # Load the MP3 file using librosa
    y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)

    # Extract MFCC features and compute mean
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=0)

    # Normalize MFCC values to have zero mean and unit variance
    mfcc_mean_norm = (mfcc_mean - np.mean(mfcc_mean)) / np.std(mfcc_mean)

    # Find frames where mean MFCC is within similar range for at least five consecutive frames
    similar_frames = []
    current_range = []
    prev_similar_mfcc = None
    for i, mfcc_val in enumerate(mfcc_mean_norm):
        if len(current_range) < 5:
            if len(current_range) == 0 or abs(mfcc_val - current_range[-1]) < threshold:
                current_range.append(mfcc_val)
            else:
                current_range = []
        elif abs(mfcc_val - current_range[-1]) < threshold:
            current_range.append(mfcc_val)
        else:
            if prev_similar_mfcc is None or abs(prev_similar_mfcc - current_range[0]) > similarity_threshold:
                similar_frames.append((i-5, i))
                prev_similar_mfcc = current_range[0]
            current_range = []

    # Plot audio waveform and mean MFCC
    fig, ax = plt.subplots(figsize=(30, 20))

    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.9)
    ax.set(title='Audio waveform')

    # Plot mean MFCC on top of waveform
    time = np.arange(len(mfcc_mean)) * len(y) / len(mfcc_mean) / sr
    ax.plot(time, mfcc_mean_norm, color='r', alpha=0.4)

    # Add vertical lines for similar frames
    for start, _ in similar_frames:
        start_time = librosa.frames_to_time(start, sr=sr) - 3
        ax.axvline(x=start_time, color='g', linestyle='--')


    ax.set_xlabel('Time (s)')
    # Save the figure as an image file
    plt.savefig('analysis_plot.png', bbox_inches='tight')
    plt.close()

    # Display the saved image
    display(Image(filename='analysis_plot.png'))

analyze_audio('https://github.com/mlmstdt/Mapping-radio/raw/main/Testradio.mp3', 0.15, 1.5)


Hermeneutics 8
import pandas as pd

df = pd.read_csv("https://github.com/mlmstdt/Mapping-radio/raw/main/audio_featuresmean.csv")
df



Hermeneutics 9
import os
import csv
import json
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import warnings
import re
import umap

csv_file_url = 'https://github.com/mlmstdt/Mapping-radio/raw/main/audio_featuresmean.csv'

# Custom sorting function
def custom_sort_key(name):
    match = re.match(r"([a-zA-Z]+)(\d+)\s(\d+)", name, re.I)
    if match:
        items = match.groups()
        return items[0], int(items[1]), int(items[2])
    else:
        return name

def plot_result(result, audio_file_info, subfolder_names, title='PCA'):
    plt.figure(figsize=(16, 16))
    plt.title(title)
    color_map = {subfolder_name: color for color, subfolder_name in zip(cm.Set2.colors, subfolder_names)}

    # Sort audio_file_info by subfolder names
    audio_file_info = sorted(audio_file_info, key=lambda x: custom_sort_key(x[1]))

    for idx, (file_name, subfolder_name) in enumerate(audio_file_info):
        plt.scatter(result[idx, 0], result[idx, 1], color=color_map[subfolder_name], label=subfolder_name, s=50, alpha=0.8)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Sort the legend using the custom_sort_key function
    sorted_legend = sorted(by_label.items(), key=lambda x: custom_sort_key(x[0]))
    plt.legend([item[1] for item in sorted_legend], [item[0] for item in sorted_legend], title='Subfolders', loc='upper right')

    plt.show()

def plot_from_csv(csv_file_url, title='PCA', selected_subfolders=None):
    data = pd.read_csv(csv_file_url)
    audio_file_info = [(row['file_name'], row['subfolder_name']) for _, row in data.iterrows()]

    # Filter audio_file_info based on selected_subfolders
    if selected_subfolders is not None:
        audio_file_info = [(file_name, subfolder_name) for file_name, subfolder_name in audio_file_info if subfolder_name in selected_subfolders]

    subfolder_names = sorted(set([subfolder_name for _, subfolder_name in audio_file_info]), key=custom_sort_key)

    # Load MFCC values directly from the columns
    mfcc_columns = [f'mfcc_{i}' for i in range(6)]
    result = np.array(data[mfcc_columns].values)
    print("Result shape:", result.shape)

    # Perform PCA on the loaded data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(result)

    plot_result(pca_result, audio_file_info, subfolder_names, title=title)
    
    return pca_result, audio_file_info, subfolder_names

def plot_from_csv_umap(csv_file_url, title='UMAP', selected_subfolders=None):
    data = pd.read_csv(csv_file_url)
    audio_file_info = [(row['file_name'], row['subfolder_name']) for _, row in data.iterrows()]

    # Filter audio_file_info based on selected_subfolders
    if selected_subfolders is not None:
                audio_file_info = [(file_name, subfolder_name) for file_name, subfolder_name in audio_file_info if subfolder_name in selected_subfolders]

    subfolder_names = sorted(set([subfolder_name for _, subfolder_name in audio_file_info]), key=custom_sort_key)

    # Load MFCC values directly from the columns
    mfcc_columns = [f'mfcc_{i}' for i in range(6)]
    result = np.array(data[mfcc_columns].values)
    print("Result shape:", result.shape)

    # Perform UMAP on the loaded data
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = reducer.fit_transform(result)

    plot_result(umap_result, audio_file_info, subfolder_names, title=title)

    return umap_result, audio_file_info, subfolder_names

selected_subfolders = ['P1 88', 'P3 88']
pca_result, audio_file_info, subfolder_names = plot_from_csv(csv_file_url, title='PCA', selected_subfolders=selected_subfolders)
umap_result, audio_file_info, subfolder_names = plot_from_csv_umap(csv_file_url, title='UMAP', selected_subfolders=selected_subfolders)


Hermeneutics 10
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import pairwise_distances
import re

csv_file_url = 'https://github.com/mlmstdt/Mapping-radio/raw/main/audio_featuresmean.csv'

# Custom sorting function
def custom_sort_key(name):
    match = re.match(r"([a-zA-Z]+)(\d+)\s(\d+)", name, re.I)
    if match:
        items = match.groups()
        return items[0], int(items[1]), int(items[2])
    else:
        return name
def plot_result(result, audio_file_info, subfolder_names, title='PCA'):
    plt.figure(figsize=(16, 16))
    plt.title(title)
    color_map = {subfolder_name: color for color, subfolder_name in zip(cm.Set2.colors, subfolder_names)}

    # Sort audio_file_info by subfolder names
    audio_file_info = sorted(audio_file_info, key=lambda x: custom_sort_key(x[1]))

    for idx, (file_name, subfolder_name) in enumerate(audio_file_info):
        plt.scatter(result[idx, 0], result[idx, 1], color=color_map[subfolder_name], label=subfolder_name, s=50, alpha=0.8)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Sort the legend using the custom_sort_key function
    sorted_legend = sorted(by_label.items(), key=lambda x: custom_sort_key(x[0]))
    plt.legend([item[1] for item in sorted_legend], [item[0] for item in sorted_legend], title='Subfolders', loc='upper right')

    plt.show()

def plot_from_csv(csv_file_url, title='PCA', selected_subfolders=None):
    data = pd.read_csv(csv_file_url)
    audio_file_info = [(row['file_name'], row['subfolder_name']) for _, row in data.iterrows()]

    # Filter audio_file_info based on selected_subfolders
    if selected_subfolders is not None:
        audio_file_info = [(file_name, subfolder_name) for file_name, subfolder_name in audio_file_info if subfolder_name in selected_subfolders]

    subfolder_names = sorted(set([subfolder_name for _, subfolder_name in audio_file_info]), key=custom_sort_key)

    # Load MFCC values directly from the columns
    mfcc_columns = [f'mfcc_{i}' for i in range(6)]
    result = np.array(data[mfcc_columns].values)
    print("Result shape:", result.shape)

    # Perform PCA on the loaded data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(result)

    plot_result(pca_result, audio_file_info, subfolder_names, title=title)
    
    return pca_result, audio_file_info, subfolder_names

def print_intra_cluster_distances(result, audio_file_info, subfolder_names, method='PCA'):
    distances = pairwise_distances(result)
    subfolder_avg_distances = {}

    for subfolder_name in subfolder_names:
        subfolder_distances = []

        for i, (_, subfolder_name_i) in enumerate(audio_file_info):
            if subfolder_name_i == subfolder_name:
                for j, (_, subfolder_name_j) in enumerate(audio_file_info):
                    if subfolder_name_j == subfolder_name and i != j:
                        subfolder_distances.append(distances[i][j])

        subfolder_avg_distances[subfolder_name] = np.mean(subfolder_distances)

    print(f"Average intra-cluster distances ({method}):")
    for subfolder_name, avg_distance in subfolder_avg_distances.items():
        print(f"{subfolder_name}: {avg_distance:.2f}")

selected_subfolders = ['P1 88', 'P1 91', 'P1 94', 'P1 98']
pca_result, audio_file_info, subfolder_names = plot_from_csv(csv_file_url, title='PCA', selected_subfolders=selected_subfolders)
print_intra_cluster_distances(pca_result, audio_file_info, subfolder_names, 'PCA')


Hermeneutics 11
get_ipython().run_line_magic('matplotlib', 'inline')
import re

csv_file_url = 'https://github.com/mlmstdt/Mapping-radio/raw/main/audio_featuresmean.csv'

# Custom sorting function
def custom_sort_key(name):
    match = re.match(r"([a-zA-Z]+)(\d+)\s(\d+)", name, re.I)
    if match:
        items = match.groups()
        return items[0], int(items[1]), int(items[2])
    else:
        return name
def plot_result(result, audio_file_info, subfolder_names, title='PCA'):
    plt.figure(figsize=(16, 16))
    plt.title(title)
    color_map = {subfolder_name: color for color, subfolder_name in zip(cm.Set2.colors, subfolder_names)}

    # Sort audio_file_info by subfolder names
    audio_file_info = sorted(audio_file_info, key=lambda x: custom_sort_key(x[1]))

    for idx, (file_name, subfolder_name) in enumerate(audio_file_info):
        plt.scatter(result[idx, 0], result[idx, 1], color=color_map[subfolder_name], label=subfolder_name, s=50, alpha=0.8)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Sort the legend using the custom_sort_key function
    sorted_legend = sorted(by_label.items(), key=lambda x: custom_sort_key(x[0]))
    plt.legend([item[1] for item in sorted_legend], [item[0] for item in sorted_legend], title='Subfolders', loc='upper right')

    plt.show()

def plot_from_csv(csv_file_url, title='PCA', selected_subfolders=None):
    data = pd.read_csv(csv_file_url)
    audio_file_info = [(row['file_name'], row['subfolder_name']) for _, row in data.iterrows()]

    # Filter audio_file_info based on selected_subfolders
    if selected_subfolders is not None:
        audio_file_info = [(file_name, subfolder_name) for file_name, subfolder_name in audio_file_info if subfolder_name in selected_subfolders]

    subfolder_names = sorted(set([subfolder_name for _, subfolder_name in audio_file_info]), key=custom_sort_key)

    # Load MFCC values directly from the columns
    mfcc_columns = [f'mfcc_{i}' for i in range(6)]
    result = np.array(data[mfcc_columns].values)
    print("Result shape:", result.shape)

    # Perform PCA on the loaded data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(result)

    plot_result(pca_result, audio_file_info, subfolder_names, title=title)
    
    return pca_result, audio_file_info, subfolder_names

def print_intra_cluster_distances(result, audio_file_info, subfolder_names, method='PCA'):
    distances = pairwise_distances(result)
    subfolder_avg_distances = {}

    for subfolder_name in subfolder_names:
        subfolder_distances = []

        for i, (_, subfolder_name_i) in enumerate(audio_file_info):
            if subfolder_name_i == subfolder_name:
                for j, (_, subfolder_name_j) in enumerate(audio_file_info):
                    if subfolder_name_j == subfolder_name and i != j:
                        subfolder_distances.append(distances[i][j])

        subfolder_avg_distances[subfolder_name] = np.mean(subfolder_distances)

    print(f"Average intra-cluster distances ({method}):")
    for subfolder_name, avg_distance in subfolder_avg_distances.items():
        print(f"{subfolder_name}: {avg_distance:.2f}")

selected_subfolders = ['P3 88', 'P3 91', 'P3 94', 'P3 98']
pca_result, audio_file_info, subfolder_names = plot_from_csv(csv_file_url, title='PCA', selected_subfolders=selected_subfolders)
print_intra_cluster_distances(pca_result, audio_file_info, subfolder_names, 'PCA')
