import numpy as np
import matplotlib.pyplot as plt


def save_figure(state_buffer, action, M, interial, force, episode, filename):
    label = ['U', 'W', 'THETA', 'q', 'speed', 'ele', 'Maero', 'Minduce', 'Mcs', 'vx', 'vz', 'x', 'z', 'lift', 'drag',
             'thrust', 'Finduce','alpha']
    length = len(state_buffer)
    t = np.linspace(0, (length - 1) * 0.008, length)
    data = list(map(list, zip(*state_buffer)))
    data1 = list(map(list, zip(*action)))
    data2 = list(map(list, zip(*M)))
    data4 = list(map(list, zip(*interial)))
    date5 = list(map(list, zip(*force)))
    fig, ax = plt.subplots(6, 3)
    for i in range(4):
        plt.subplot(6, 3, i + 1)
        plt.plot(t, data[i])
        plt.title(label[i])
    for i in range(4, 6):
        plt.subplot(6, 3, i + 1)
        plt.plot(t, data1[i - 4])
        plt.title(label[i])
    for i in range(6, 9):
        plt.subplot(6, 3, i + 1)
        plt.plot(t, data2[i - 6])
        plt.title(label[i])
    for i in range(9, 13):
        plt.subplot(6, 3, i + 1)
        plt.plot(t, data4[i - 9])
        plt.title(label[i])
    for i in range(13, 18):
        plt.subplot(6, 3, i + 1)
        plt.plot(t, date5[i - 13])
        plt.title(label[i])
    plt.tight_layout()
    plt.savefig('{}/{}episode.jpg'.format(filename, episode), bbox_inches='tight')
