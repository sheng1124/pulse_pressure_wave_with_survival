import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
import tensorflow as tf
np.random.seed(2021)
tf.random.set_seed(2021)

def load_data():
  path = 'data'
  datalist=[]
  for dirpath, dirname, filelist in os.walk(path):
    for file in filelist:
      filepath = os.path.join(dirpath, file)
      if '.txt' in filepath and('DeepPulse' in filepath or 'MediumPulse' in filepath or 'ShallowPulse' in filepath):
        if 'DeepPulse' in filepath:
          pluse = 'DeepPulse'
        elif 'MediumPulse' in filepath:
          pluse = 'MediumPulse'
        else:
          pluse = 'ShallowPulse'
        id_ = os.path.dirname(dirpath)
        id_ = os.path.basename(id_)
        date = os.path.basename(dirpath)
        data = load_file(filepath, pluse)
        if id_ in {'21114379', '20684466'}:
          alive = 'DarkGreen'
        elif id_ in {'22256880', '21534157'}:
          alive = 'Gray'
        else:
          alive = 'Red'
        datalist.append({'id':id_, 'date':date, 'pluse':pluse, 'data': data, 'alive':alive, 'filepath': filepath})
  return datalist

def load_file(filepath, pluse):
  fin = open(filepath, 'r', newline='')
  fin.readline()
  ylist = []
  for line in fin:
    ylist.append(int(line))
  xlist = [e/500 for e in range(1, len(ylist)+1)]
  data = pd.DataFrame(ylist, index=xlist, columns=[pluse])
  return data

def out_table():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  tdd = td[['id', 'date', 'pluse', 'alive']]
  tdd = tdd.reset_index()
  plt.figure('123')           # 視窗名稱
  ax = plt.axes(frame_on=False)# 不要額外框線
  ax.xaxis.set_visible(False)  # 隱藏X軸刻度線
  ax.yaxis.set_visible(False)  # 隱藏Y軸刻度線
  pd.plotting.table(ax, tdd, colLabels ='', loc='center')

def diff(td):
  for row in td.iloc():
    for i in range(1, len(row['data'])):
      #input(str(row['data']))
      row['data'].iloc[i-1] = row['data'].iloc[i] - row['data'].iloc[i-1]

def find_lamda(da):
  dlist = da.values.tolist()
  dlist = [e[0] for e in dlist]
  yminid = 0
  ymin = dlist[0]
  ymaxid = 0
  ymax = dlist[0]
  for i in range(len(dlist)):
    if dlist[i] >= ymax:
      ymax = dlist[i]
      ymaxid = i
    if dlist[i] <= ymin:
      ymin = dlist[i]
      yminid = i
  print(ymax, ymin)
  ymid = (ymax + ymin) / 2
  dlist = [e - ymid for e in dlist]

  for e in da.iloc():
    e[0] -= ymid
  
  start = yminid if yminid < ymaxid else ymaxid

  for i in range(start, len(dlist)-1):
    if dlist[i] == 0 or dlist[i] * dlist[i+1] < 0:
      lam = i - start
      print('start:', start/500, ' 1/4 l:',lam/500, ' lamda: ', 4*lam/500)
      return lam

def find_max(td):
  ymax = 0
  for row in td.iloc():
    dl = row['data'].values.tolist()
    dl = [e[0] for e in dl]
    ymax2 = max(dl)
    if ymax2 >= ymax:
      ymax = ymax2
  return ymax

def normalize(td, ymax):
  for row in td.iloc():
    row['data'] = row['data'].astype(float)
    for e in row['data'].iloc():
      e[0] = e[0] / ymax
  return td

def get_data(td):
  x_alive, y_alive, x_dead, y_dead, x_unknown = ([],[],[],[],[])
  for row in td.iloc():
    dl = row['data'].values.tolist()
    dl = [e[0] for e in dl]
    if row['alive'] != 'Gray': #test
      if row['alive'] == 'DarkGreen':
        y = [1.0, 0.0]
        x_alive.append(dl)
        y_alive.append(y)
      else:
        y = [0.0, 1.0]
        x_dead.append(dl)
        y_dead.append(y)
    else:
      x_unknown.append(dl)
  return ([x_alive, y_alive], [x_dead, y_dead], [x_unknown])

def get_data_2D_L(td):
  alive_data, dead_data, unknown_data = get_data(td)
  x_alive, y_alive =  alive_data
  x_dead, y_dead = dead_data
  x_unknown = unknown_data[0]
  x_all=[]
  for ds in [x_alive, x_dead, x_unknown]:
    nds = []
    for unit in ds: # unit : 2500 x
      p = plot_2D_wave(unit)
      z = Image.open(p)
      z = z.convert('L')
      z = np.asarray(z)
      z = z.reshape(z.shape[0],z.shape[1],1)
      nds.append(z)
    x_all.append(nds)
  x_alive, x_dead, x_unknown = x_all
  return ([x_alive, y_alive], [x_dead, y_dead], [x_unknown])

def get_data_2D_rgb(td):
  alive_data, dead_data, unknown_data = get_data(td)
  x_alive, y_alive =  alive_data
  x_dead, y_dead = dead_data
  x_unknown = unknown_data[0]
  x_all=[]
  for ds in [x_alive, x_dead, x_unknown]:
    nds = []
    for unit in ds: # unit : 2500 x
      p = plot_2D_wave(unit)
      z = Image.open(p)
      z = np.asarray(z)
      nds.append(z)
    x_all.append(nds)
  x_alive, x_dead, x_unknown = x_all
  return ([x_alive, y_alive], [x_dead, y_dead], [x_unknown])

def plot_2D_wave(unit):
  path = 't.jpg'
  length = len(unit)
  fig = plt.figure(figsize=(6,6))
  plt.axis('off')
  plt.axis('square')
  plt.xlim(0, length)
  plt.ylim(0, length)
  ypoints=[]
  for x in unit:
    y = int(length * x)
    y = length - 1 if y == length else y
    ypoints.append(y)
  plt.plot(range(length), ypoints)
  plt.savefig(path, bbox_inches='tight', pad_inches=0)
  plt.close(fig)
  return path

def get_train_test_data_2d_rgb(td):
  alive_data, dead_data, unknown_data = get_data_2D_rgb(td)
  return get_train_test(alive_data, dead_data)

def get_train_test_data(td):
  alive_data, dead_data, unknown_data = get_data(td)
  return get_train_test(alive_data, dead_data)

def get_train_test_data_2d(td):
  alive_data, dead_data, unknown_data = get_data_2D_L(td)
  return get_train_test(alive_data, dead_data)

def get_train_test(alive_data, dead_data):
  x_test = [alive_data[0][0], dead_data[0][0]] #alive_data[0] -> x_alive
  y_test = [alive_data[1][0], dead_data[1][0]]
  x_train = alive_data[0][1:] + dead_data[0][1:]
  y_train = alive_data[1][1:] + dead_data[1][1:]
  return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def pt_shape(x_train, y_train, x_test, y_test):
  print('x_train_shape:', x_train.shape)
  print('y_train_shape:',y_train.shape)
  print('x_test_shape:',x_test.shape)
  print('y_test_shape:',y_test.shape)

def k_fold_test(td, get_data_func):#11+2=13
  alive_data, dead_data, unknown_data = get_data_func(td)#rgb L
  alive=[]
  for i in range(len(alive_data[0])):
    alive.append([alive_data[0][i], alive_data[1][i]])
  dead=[]
  for i in range(len(dead_data[0])):
    dead.append([dead_data[0][i], dead_data[1][i]])
  np.random.shuffle(alive)
  np.random.shuffle(dead)
  a = [[],[],[],[],[],[],[]]
  nal= alive + dead[::-1]
  for i in range(7):
    if i==6:
      x_test = [nal[i][0]]
      y_test = [nal[i][1]]
      x_train = [x[0] for x in nal if id(x[0]) != id(nal[i][0])]
      y_train = [y[1] for y in nal if id(y[1]) != id(nal[i][1])]
      a[i] = [np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)]
      break
    else:
      x_test = [nal[i][0], dead[i][0]]
      y_test = [nal[i][1], dead[i][1]]
      t = (id(nal[i][0]), id(dead[i][0]))
      x_train = np.array([x[0] for x in nal if not id(x[0]) in t])
      y_train = np.array([y[1] for y in nal if not id(y[0]) in t])
      a[i] = [np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)]
  return a
    #test = [alive[i],dead[i]]   #A1 A2 A3 A4 A5        dead
    #a[i] = alive[i]             #B1 B2 A5 A4 A3 A2 A1  nal
  

def main():
  main_kf_res()

def main_kf_res():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  dp = td[td.pluse == 'DeepPulse']
  mp = td[td.pluse == 'MediumPulse']
  sp = td[td.pluse == 'ShallowPulse']
  for ddf in [dp]:#dp, mp, sp
    normalize(ddf, find_max(ddf))
    ksets = k_fold_test(ddf, get_data_2D_rgb)
    te_list=[]
    tr_list=[]
    for k in ksets:
      x_train, y_train, x_test, y_test = k
      pt_shape(x_train, y_train, x_test, y_test)
      res_model = do_resnet(x_train, y_train)
      tr,te = test_model_acc(res_model, x_train, y_train, x_test, y_test)
      te_list.append(te)
      tr_list.append(tr)
      model_predict(res_model, x_train, y_train, x_test, y_test)
    print('Avg train Acc: ', np.mean(te_list))
    print('Avg test Acc: ', np.mean(tr_list))

def main_kf_cnn():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  dp = td[td.pluse == 'DeepPulse']
  mp = td[td.pluse == 'MediumPulse']
  sp = td[td.pluse == 'ShallowPulse']
  for ddf in [dp]:#dp, mp, sp
    normalize(ddf, find_max(ddf))
    ksets = k_fold_test(ddf, get_data_2D_L)
    te_list=[]
    tr_list=[]
    for k in ksets:
      x_train, y_train, x_test, y_test = k
      pt_shape(x_train, y_train, x_test, y_test)
      cnn_model = do_cnn(x_train, y_train)
      tr,te = test_model_acc(cnn_model, x_train, y_train, x_test, y_test)
      te_list.append(te)
      tr_list.append(tr)
      model_predict(cnn_model, x_train, y_train, x_test, y_test)
    print('Avg train Acc: ', np.mean(tr_list))
    print('Avg test Acc: ', np.mean(te_list))
    #print(te_list)

def main_dnn():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  dp = td[td.pluse == 'DeepPulse']
  mp = td[td.pluse == 'MediumPulse']
  sp = td[td.pluse == 'ShallowPulse']
  for ddf in [dp]:#dp, mp, sp
    normalize(ddf, find_max(ddf))
    x_train, y_train, x_test, y_test = get_train_test_data(ddf)
    pt_shape(x_train, y_train, x_test, y_test)
    dnn_model = do_dnn(x_train, y_train)
    test_model_acc(dnn_model, x_train, y_train, x_test, y_test)
    model_predict(dnn_model, x_train, y_train, x_test, y_test)

def main_cnn():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  dp = td[td.pluse == 'DeepPulse']
  mp = td[td.pluse == 'MediumPulse']
  sp = td[td.pluse == 'ShallowPulse']
  for ddf in [dp]:#dp, mp, sp
    normalize(ddf, find_max(ddf))
    x_train, y_train, x_test, y_test = get_train_test_data_2d(ddf)
    pt_shape(x_train, y_train, x_test, y_test)
    cnn_model = do_cnn(x_train, y_train)
    test_model_acc(cnn_model, x_train, y_train, x_test, y_test)
    model_predict(cnn_model, x_train, y_train, x_test, y_test)

def main_resnet():
  td = pd.DataFrame(load_data())
  td = td.sort_values(['id', 'date'],ascending=False)
  dp = td[td.pluse == 'DeepPulse']
  mp = td[td.pluse == 'MediumPulse']
  sp = td[td.pluse == 'ShallowPulse']
  for ddf in [dp]:#dp, mp, sp
    normalize(ddf, find_max(ddf))
    x_train, y_train, x_test, y_test = get_train_test_data_2d_rgb(ddf)
    pt_shape(x_train, y_train, x_test, y_test)
    res_model = do_resnet(x_train, y_train)
    test_model_acc(res_model, x_train, y_train, x_test, y_test)
    model_predict(res_model, x_train, y_train, x_test, y_test)

def test_model_acc(model, x_train, y_train, x_test, y_test):
  result_train = model.evaluate(x_train, y_train)
  print('\nTrain Acc:\n', result_train[1])
  result_test = model.evaluate(x_test, y_test)
  print('\nTest Acc:\n', result_test[1])
  return result_train[1], result_test[1]

def model_predict(model, x_train, y_train, x_test, y_test):
  pred = model.predict(x_train)
  print('Train predict:\n', pred[0:2] ,'\n', pred[-3:])
  print('answer:\n', y_train[0:2] ,'\n', y_train[-3:])
  pred = model.predict(x_test)
  show_data_x(x_test)
  print('Test predict:\n', pred)
  print('answer:\n', y_test)

def show_data_x(x_set):
  for i in range(len(x_set)):
    fig = plt.figure()
    plt.title('test x{}'.format(i))
    if x_set[i].shape[2] == 1:
      z = x_set[i].shape[0]
      xb = x_set[i].reshape(z, z)
      plt.imshow(xb, cmap='gray', vmin=0, vmax=255)
    else:
      plt.imshow(x_set[i])
    plt.show()
    #plt.close(fig)


def printfig(td):
  last_id = ''
  for row in td.iloc():
    new_id = row['id']
    if new_id != last_id:
      print('========================= new id ==============================')
      last_id = new_id
    print('id:{}, date:{}, pluse:{}, alive:{}'.format(row['id'], row['date'], row['pluse'], row['alive']))
    row['data'].plot(color=row['alive'],xlabel = 'sec',xlim=[0.0, 5.0],ylim=[0.0, 1.0], ylabel='PI',title='date:{} id:{}'.format(row['date'], row['id']))
    #row['data'].plot(color=row['alive'],xlabel = 'sec',xlim=[0.0, 5.0],ylim=[-50.0, 50.0], ylabel='PI',title='date:{} id:{}'.format(row['date'], row['id']))
    plt.show()
    print()
  print()
  print()

def outfig(td):
  last_id = ''
  for row in td.iloc():
    new_id = row['id']
    if new_id != last_id:
      print('========================= new id ==============================')
      last_id = new_id
    print('id:{}, date:{}, pluse:{}, alive:{}'.format(row['id'], row['date'], row['pluse'], row['alive']))
    row['data'].plot(color=row['alive'],xlabel = 'sec',xlim=[0.0, 5.0], ylabel='PI',title='date:{} id:{}'.format(row['date'], row['id']))
    plt.savefig(row['filepath'][:-4]+'.png')
    plt.show()
    print()
  print()
  print()

def do_dnn(x_train, y_train):
  model = Sequential()
  model.add(Dense(input_dim = x_train.shape[1], units=400, activation='relu'))
  model.add(Dense(units=400, activation='relu'))
  model.add(Dense(units=2, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=3, epochs=20)
  return model

def do_cnn(x_train, y_train):
  width, height, depth = x_train.shape[1:]
  model = Sequential()
  model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(3,3), filters=25))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(kernel_size=(3,3), filters=50))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(units=100, activation='relu'))
  model.add(Dense(units=2, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=3, epochs=40)
  return model

def do_resnet(x_train, y_train):
  width, height, depth = x_train.shape[1:]
  model = Sequential()
  model.add(ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=(width, height, depth)))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(units=2, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=3, epochs=20)
  return model


main()
