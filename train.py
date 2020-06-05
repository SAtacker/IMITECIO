import c3d
import os

path = os.path.abspath(os.getcwd())
path += r"\Human Eva\S1\Mocap_Data\Box_1.c3d"
with open(path, 'rb') as handle:
    reader = c3d.Reader(handle)
    for i, (points, analog) in enumerate(reader.read_frames()):
        print('Frame {}: {}'.format(i, points.round(2)))