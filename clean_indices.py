import pickle

with open('/home/konovalyuk/data/isic/indices_isic2019.pkl','rb') as f:
    indices = pickle.load(f)
    print(indices['trainIndCV'][0], '\n')
    print(indices['trainIndCV'][1], '\n')
    print(indices['trainIndCV'][2], '\n')
    print(indices['trainIndCV'][3], '\n')
    print(indices['trainIndCV'][4])