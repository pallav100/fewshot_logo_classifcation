import random
from protonet import ProtoNet

from prototypical_loss import my_prototypical_loss as loss_fn
import numpy as np
import torch
from PIL import Image
import os
import pickle
def get_embd(x, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
#    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     x, y = x.to(device), y.to(device)
    out = model(x)
    return (out)
model = ProtoNet()
model_path = '/home/pallav_soni/pro/output/best_model.pth'
model.load_state_dict(torch.load(model_path))

def load_img(path):
    x = Image.open(path).convert('RGB')
    x = x.resize((28, 28))
    shape = 3, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
    x = torch.unsqueeze(x,0)
    return x
#img = load_img('/home/pallav_soni/dumm.jpeg')
#img = torch.unsqueeze(img, 0)
#print(img.size())
#embd = get_embd(img,model)
means_path = '/home/pallav_soni/pro/model/prev_latest_means.pt'

#log_query = loss_fn(embd,means_path)

#print(log_query[1].item())
with open('/home/pallav_soni/pro/model/classes.pkl', 'rb') as f:
	classlist = pickle.load(f)
#inv_classlist = {v: k for k, v in classlist.items()}
sorted_x = sorted(classlist.items(), key=lambda kv: kv[1])
for item  in sorted_x:
    print("{} : {}".format(item[1],item[0]))
path = '/home/pallav_soni/oplogo/crops_resized_300/'
#print(sorted_x[1][0])
def new():
	imgs=[]
	ground = []
	labels = []
	j=0
	for clas in classlist.keys():
		j+=1
		clpath = path+clas+'/'
		for i in range(3):
			x= random.choice([x for x in os.listdir(clpath) if os.path.isfile(os.path.join(path+clas+'/',x))])
			file = os.path.join(clpath, x)
			imgs.append(file)
			ground.append(clas)
	for l in range(len(imgs)):
			jpg = load_img(imgs[l])
			emd = get_embd(jpg,model)
			lbl = loss_fn(emd,means_path)
			labels.append(lbl[1].item())
	acc=0
	for i in range(len(labels)):
		if(sorted_x[labels[i]][0]==ground[i]):
			acc+=1
	return acc/len(imgs)
lbls = new()
print(lbls)
#for i in range(len(lbls)):
#	if(lbls[i] ==
import matplotlib.patheffects as PathEffects
import seaborn as sns
import torch
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import numpy as np
time_start = time.time()
means = torch.load('/home/pallav_soni/pro/model/prev_latest_means.pt')
#img1 = load_img('/home/pallav_soni/oplogo/crops_resized_28/pepsi/pepsi_1.jpg')
#img2 = load_img('/home/pallav_soni/oplogo/crops_resized_28/lg/lg_1.jpg')

#embd1 = get_embd(img1,model)
#embd2 = get_embd(img2,model)
#means = torch.cat([means,embd1],dim=0)
#means = torch.cat([means,embd2],dim=0)
print(means.size())
meansnp = means.detach()
cl = [i for i in range(means.size()[0])]
#cl.append(88)
#cl.append(112)
cl = np.array(cl)
fashion_tsne = TSNE().fit_transform(meansnp)

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=7)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig('/home/pallav_soni/foofl.png')
    return f, ax, sc, txts


fashion_scatter(fashion_tsne,cl)
