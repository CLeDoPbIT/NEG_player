import os
import threading
import time
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog

from tkinter import ttk
from ttkthemes import themed_tk as tk

from mutagen.mp3 import MP3
from pygame import mixer

import soundfile as sf
import utils
import torch
import numpy as np
device = 'cpu'

root = tk.ThemedTk()
root.get_themes()
root.set_theme("radiance")


statusbar = ttk.Label(root, text="Welcome to Melody", relief=SUNKEN, anchor=W, font='Times 10 italic')
statusbar.pack(side=BOTTOM, fill=X)

menubar = Menu(root)
root.config(menu=menubar)

subMenu = Menu(menubar, tearoff=0)

playlist = []


def browse_file():
    global filename_path
    filename_path = filedialog.askopenfilename()
    add_to_playlist(filename_path)

    mixer.music.queue(filename_path)


def add_to_playlist(filename):
    filename = os.path.basename(filename)
    index = 0
    playlistbox.insert(index, filename)
    playlist.insert(index, filename_path)
    index += 1


def separate_audio():
    selected_song = playlistbox.curselection()
    selected_song = int(selected_song[0])
    play_it = playlist[selected_song]

    consts = utils.Constants("speech")

    model = torch.load('model-9', map_location='cpu').to(device)

    audio = sf.read(play_it)

    spec_abs, spec = utils.aud2spec(audio, consts)

    test_mask = []
    test_batch_Y = spec_abs
    test_batch_Y = torch.from_numpy(test_batch_Y).to(device).float()
    batch_mask = model(test_batch_Y)
    test_mask.append(batch_mask.cpu().data.numpy())

    test_mask = np.concatenate(test_mask, axis=0).squeeze()
    B_test_prediction_spectrogram = spec[:, 0] * test_mask
    X_test_prediction_spectrogram = spec[:, 0] * (1 - test_mask)

    B_pred_audio = utils.spec2aud(B_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)
    X_pred_audio = utils.spec2aud(X_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)

    sf.write("B_test_pred.wav", B_pred_audio, consts.SR)
    sf.write("X_test_pred.wav", X_pred_audio, consts.SR)

    index = 0
    playlistbox.insert(index, "B_test_pred")
    playlist.insert(index, "B_test_pred.wav")
    index += 1
    playlistbox.insert(index, "X_test_pred")
    playlist.insert(index, "X_test_pred.wav")
    index += 1



menubar.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Open", command=browse_file)
subMenu.add_command(label="Exit", command=root.destroy)


def about_us():
    tkinter.messagebox.showinfo('About Melody', 'This is a music player build using Python Tkinter by @attreyabhatt')


subMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=subMenu)
subMenu.add_command(label="About Us", command=about_us)

mixer.init(frequency=16000)

root.title("Melody")

leftframe = Frame(root)
leftframe.pack(side=LEFT, padx=30, pady=30)

playlistbox = Listbox(leftframe)
playlistbox.pack()

addBtn = ttk.Button(leftframe, text="+ Add", command=browse_file)
addBtn.pack(side=LEFT)

separeteBtn = ttk.Button(leftframe, text="Separete", command=separate_audio)
separeteBtn.pack(side=LEFT)


def del_song():
    selected_song = playlistbox.curselection()
    selected_song = int(selected_song[0])
    playlistbox.delete(selected_song)
    playlist.pop(selected_song)


delBtn = ttk.Button(leftframe, text="- Del", command=del_song)
delBtn.pack(side=LEFT)

rightframe = Frame(root)
rightframe.pack(pady=30)

topframe = Frame(rightframe)
topframe.pack()

lengthlabel = ttk.Label(topframe, text='Total Length : --:--')
lengthlabel.pack(pady=5)

currenttimelabel = ttk.Label(topframe, text='Current Time : --:--', relief=GROOVE)
currenttimelabel.pack()


def show_details(play_song):
    file_data = os.path.splitext(play_song)

    if file_data[1] == '.mp3':
        audio = MP3(play_song)
        total_length = audio.info.length
    else:
        a = mixer.Sound(play_song)
        total_length = a.get_length()

    # div - total_length/60, mod - total_length % 60
    mins, secs = divmod(total_length, 60)
    mins = round(mins)
    secs = round(secs)
    timeformat = '{:02d}:{:02d}'.format(mins, secs)
    lengthlabel['text'] = "Total Length" + ' - ' + timeformat

    t1 = threading.Thread(target=start_count, args=(total_length,))
    t1.start()


def start_count(t):
    global paused
    current_time = 0
    while current_time <= t and mixer.music.get_busy():
        if paused:
            continue
        else:
            mins, secs = divmod(current_time, 60)
            mins = round(mins)
            secs = round(secs)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            currenttimelabel['text'] = "Current Time" + ' - ' + timeformat
            time.sleep(1)
            current_time += 1


def play_music():
    global paused

    if paused:
        mixer.music.unpause()
        statusbar['text'] = "Music Resumed"
        paused = FALSE
    else:
        try:
            stop_music()
            time.sleep(1)
            selected_song = playlistbox.curselection()
            selected_song = int(selected_song[0])
            play_it = playlist[selected_song]
            mixer.music.load(play_it)
            mixer.music.play()
            statusbar['text'] = "Playing music" + ' - ' + os.path.basename(play_it)
            show_details(play_it)
        except:
            tkinter.messagebox.showerror('File not found', 'Melody could not find the file. Please check again.')


def stop_music():
    mixer.music.stop()
    statusbar['text'] = "Music Stopped"


paused = FALSE


def pause_music():
    global paused
    paused = TRUE
    mixer.music.pause()
    statusbar['text'] = "Music Paused"


def set_vol(val):
    volume = float(val) / 100
    mixer.music.set_volume(volume)

muted = FALSE


def mute_music():
    global muted
    if muted:  # Unmute the music
        mixer.music.set_volume(0.7)
        volumeBtn.configure(image=volumePhoto)
        scale.set(70)
        muted = FALSE
    else:  # mute the music
        mixer.music.set_volume(0)
        volumeBtn.configure(image=mutePhoto)
        scale.set(0)
        muted = TRUE




middleframe = Frame(rightframe)
middleframe.pack(pady=30, padx=30)

playPhoto = PhotoImage(file='images/play.png')
playBtn = ttk.Button(middleframe, image=playPhoto, command=play_music)
playBtn.grid(row=0, column=0, padx=10)

stopPhoto = PhotoImage(file='images/stop.png')
stopBtn = ttk.Button(middleframe, image=stopPhoto, command=stop_music)
stopBtn.grid(row=0, column=1, padx=10)

pausePhoto = PhotoImage(file='images/pause.png')
pauseBtn = ttk.Button(middleframe, image=pausePhoto, command=pause_music)
pauseBtn.grid(row=0, column=2, padx=10)

bottomframe = Frame(rightframe)
bottomframe.pack()

mutePhoto = PhotoImage(file='images/mute.png')
volumePhoto = PhotoImage(file='images/volume.png')
volumeBtn = ttk.Button(bottomframe, image=volumePhoto, command=mute_music)
volumeBtn.grid(row=0, column=1)

scale = ttk.Scale(bottomframe, from_=0, to=100, orient=HORIZONTAL, command=set_vol)
scale.set(70)  # implement the default value of scale when music player starts
mixer.music.set_volume(0.7)
scale.grid(row=0, column=2, pady=15, padx=30)


def on_closing():
    stop_music()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
