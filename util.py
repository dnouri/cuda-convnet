import re
import cPickle
import os
import numpy as n
from math import sqrt

import gzip
import zipfile

VENDOR_ID_REGEX = re.compile('^vendor_id\s+: (\S+)')
GPU_LOCK_NO_SCRIPT = -2
GPU_LOCK_NO_LOCK = -1

try:
    import magic
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
except ImportError: # no magic module
    ms = None

def get_gpu_lock():
    import imp
    lock_script_path = '/u/murray/bin/gpu_lock.py'
    if os.path.exists(lock_script_path):
        locker = imp.load_source("", lock_script_path)
        return locker.obtain_lock_id()
    return GPU_LOCK_NO_SCRIPT

def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    if ms is not None and ms.file(filename).startswith('gzip'):
        fo = gzip.open(filename, 'rb')
        dict = cPickle.load(fo)
    elif ms is not None and ms.file(filename).startswith('Zip'):
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    
    fo.close()
    return dict

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def is_intel_machine():
    f = open('/proc/cpuinfo')
    for line in f:
        m = VENDOR_ID_REGEX.match(line)
        if m:
            f.close()
            return m.group(1) == 'GenuineIntel'
    f.close()
    return False

def get_cpu():
    if is_intel_machine():
        return 'intel'
    return 'amd'

def splice_rows(r1, r2):
    return n.r_[r1,r2]
    
def splice_cols(c1, c2):
    return n.c_[c1,c2]
    
def grid_to_matrix(m, sq_size):
    img_size = int(sqrt(m.shape[1]))
    m = m.reshape(m.shape[0], img_size, img_size).swapaxes(0,2) # trans, cols first
    #m = reduce(splice_rows, [m[:,t:t+sq_size,:] for t in range(0, m.shape[1], sq_size)])
    m = n.concatenate(n.split(m, m.shape[1]/sq_size, axis=1))
    m = m.reshape(m.shape[0]/sq_size,sq_size*sq_size,m.shape[2])
    m = m.swapaxes(0,2).swapaxes(1,2)
    return m.reshape(m.shape[0]*m.shape[1],m.shape[2])
    
def grid_to_matrix_color(m, sq_size):
    return grid_to_matrix(m.reshape(m.shape[0] * 3,m.shape[1]/3), sq_size)
    
def matrix_to_grid(m, img_size):
    sq_size = int(sqrt(m.shape[1]))
    num_imgs = (m.shape[0]*sq_size*sq_size) / (img_size*img_size) 
    m = m.reshape(num_imgs, img_size*img_size/sq_size, sq_size)
    #m = reduce(splice_cols, [m[:,t:t+img_size,:] for t in range(0, m.shape[1], img_size)])
    m = n.concatenate(n.split(m, m.shape[1]/img_size, axis=1), axis=2)
    m = m.swapaxes(1,2).reshape(num_imgs, img_size*img_size)
    return m
    
def matrix_to_grid_color(m, img_size):
    m = matrix_to_grid(m, img_size)
    return m.reshape(m.shape[0]/3, m.shape[1]*3)
    
def localavg_blowup(m, avg_size):
    if len(m.shape) == 1:
        m = m.reshape(1, m.shape[0])
    img_size = int(sqrt(m.shape[1]))
    matrix = n.tile(n.c_[grid_to_matrix(m, avg_size).mean(axis=1)], (1, avg_size*avg_size))
    return matrix_to_grid(matrix, img_size)

def localavg_blowup_color(m, avg_size):
    if len(m.shape) == 1:
        m = m.reshape(1, m.shape[0])
    m = localavg_blowup(m.reshape(m.shape[0]*3,m.shape[1]/3), avg_size)
    return m.reshape(m.shape[0]/3,m.shape[1]*3)

def localavg(m, avg_size):
    if len(m.shape) == 1:
        m = m.reshape(1, m.shape[0])
    img_size = int(sqrt(m.shape[1]))
    matrix = n.c_[grid_to_matrix(m, avg_size).mean(axis=1)]
    return matrix_to_grid(matrix, img_size/avg_size)

def localavg_color(m, avg_size):
    if len(m.shape) == 1:
        m = m.reshape(1, m.shape[0])
    m = localavg(m.reshape(m.shape[0]*3,m.shape[1]/3), avg_size)
    return m.reshape(m.shape[0]/3,m.shape[1]*3)

# eye pos on subsampled plane
# 8x8 center high-res, 4px border subsampled to 2px
# for a total of 112 (* 3) pixels
def get_eye_view(data, data_subs2, eye_pos):
    img_size = int(sqrt(data.shape[1]))
    subs_img_size = int(sqrt(data_subs2.shape[1]))
    y, x = eye_pos
    data = data.reshape(data.shape[0], img_size, img_size)
    data_subs2 = data_subs2.reshape(data_subs2.shape[0], subs_img_size, subs_img_size)
    
    border = n.c_[data_subs2[:,y:y+2,x:x+8].reshape(data_subs2.shape[0], 8*2),
                  data_subs2[:,y+2:y+8,x:x+2].reshape(data_subs2.shape[0], 6*2),
                  data_subs2[:,y+2:y+8,x+6:x+8].reshape(data_subs2.shape[0], 6*2),
                  data_subs2[:,y+6:y+8,x+2:x+6].reshape(data_subs2.shape[0], 4*2)]
    x *= 2
    y *= 2
    return n.c_[data[:,(y+4):y+12,(x+4):x+12].reshape(data.shape[0], 8*8), border]

def get_eye_view_color(data, data_subs2, eye_pos):
    view = get_eye_view(data.reshape(3*data.shape[0],data.shape[1]/3),
                        data_subs2.reshape(3*data_subs2.shape[0],data_subs2.shape[1]/3),
                        eye_pos)
    return view.reshape(view.shape[0]/3, view.shape[1]*3)

def get_eye_plottable(eye_view):
    eye_p = n.zeros((eye_view.shape[0], 16, 16), dtype=eye_view.dtype)
    eye_view_center = eye_view[:, n.r_[n.arange(0,64)]].reshape(eye_view.shape[0], 8, 8)
    
    eye_p[:, 4:12, 4:12] = eye_view_center
    bar_top = eye_view[:, n.r_[n.arange(64,64+8*2)]].reshape(eye_view.shape[0], 2, 8).repeat(2,axis=1).repeat(2,axis=2)
    bar_left = eye_view[:, n.r_[n.arange(80,80+6*2)]].reshape(eye_view.shape[0], 6, 2).repeat(2,axis=1).repeat(2,axis=2)
    bar_right = eye_view[:, n.r_[n.arange(92,92+6*2)]].reshape(eye_view.shape[0], 6, 2).repeat(2,axis=1).repeat(2,axis=2)
    bar_bottom = eye_view[:, n.r_[n.arange(104,104+4*2)]].reshape(eye_view.shape[0], 2, 4).repeat(2,axis=1).repeat(2,axis=2)
    
    eye_p[:, 0:4, 0:16] = bar_top
    eye_p[:, 4:16, 0:4] = bar_left
    eye_p[:, 4:16, 12:16] = bar_right
    eye_p[:, 12:16, 4:12] = bar_bottom
    
    return eye_p.reshape(eye_p.shape[0], 256)
    
def get_eye_plottable_color(eye_view_color):
    eye_p = get_eye_plottable(eye_view_color.reshape(eye_view_color.shape[0]*3, eye_view_color.shape[1]/3))
    return eye_p.reshape(eye_p.shape[0]/3, eye_p.shape[1]*3)