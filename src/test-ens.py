import numpy as np, pandas as pd, pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

kelly_colors = ['#F2F3F4', '#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']
kelly_colors = [c + "FF" for c in kelly_colors]

x = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-x2.csv").values
y = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-y2.csv").values

colors = []

modelfile = r"C:\Users\Brian\Desktop\Programming\datathon\dataset\model.sav"
model = pickle.load(open(modelfile, 'rb'))

y_pred = model.predict(x)


y_max = [np.argmax(arr) for arr in y_pred]


for i in range(len(y_max)):
    color = kelly_colors[y_max[i]]
    colors.append(color)


pca = PCA(n_components=2, svd_solver='randomized', random_state=1)
pca2d = pca.fit_transform(y)

scatter = plt.scatter([x[0] for x in pca2d], [y[1] for y in pca2d], c=colors)

plt.show()
