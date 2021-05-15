import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from sklearn.utils import shuffle
from torch.nn import parameter
from PIL import Image
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression


class neuralnet(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=64, num_layers=0, p_dropout=0.0, device='cpu', softmax=False):
        super().__init__()
        self.layers = nn.Sequential()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        prev_size = input_dim
        this_size = hidden_dim
        for i in range(num_layers):
            self.layers.add_module('layer{}'.format(i), nn.Linear(prev_size, this_size))
            self.layers.add_module('activation{}'.format(i), nn.ReLU())
            self.layers.add_module('DropOut{}'.format(i), nn.Dropout(p=p_dropout))
            prev_size = hidden_dim
        self.layers.add_module('outputLayer', nn.Linear(prev_size, output_dim))
        if softmax:
            self.layers.add_module('softmax', nn.Softmax(1))
        self.to(self.device)

    def forward(self, input):
        return self.layers(input)


def testing(model, dataset):
    generator = torch.utils.data.DataLoader(dataset, batch_size=16)

    predict = []
    real = []
    for data, target in tqdm(generator, leave=False):
        data = data.view([-1, 784]).to(model.device)
        target = target.to(model.device)

        predict.extend((torch.argmax(model(data), dim=-1)).cpu().numpy().tolist())
        real.extend(target.cpu().numpy().tolist())

    return np.mean(np.array(predict) == np.array(real))


def training(model, dataset, loss_func, optimizer, epochs):
    for epoch in tqdm(range(epochs), leave=False):
        generator = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True)

        for data, target in tqdm(generator, leave=False):
            optimizer.zero_grad()
            data = data.view([-1, 784]).to(model.device)
            target = target.to(model.device)

            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()


def predictt(model, dataset):
    dataset = dataset.view([-1, 784]).to(model.device)
    print(dataset)
    return ((torch.argmax(model(dataset), dim=-1)).cpu().numpy().tolist())


mnist_train = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor())

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name())

######################## TRAINING ########################
"""valid = KFold(5)
val_of_batch = valid.get_n_splits(mnist_train)

hyperparametres = ParameterGrid({'lr': [0.001, 0.01],
					'val_layers': [0, 1, 2],
					'hidden_neur': [8,16,64],
					'p': [0.3, 0.7],
					'l2': [0.0001],
					'epochs': [5, 10]})

X_TRAIN = mnist_train.transform(mnist_train.data.numpy()).transpose(0,1)
Y_TRAIN = mnist_train.targets.data"""

"""scores = []
model = neuralnet(hidden_dim= 64, p_dropout=0.3, num_layers = 1, softmax = True)
model.train()
training(model=model, dataset=mnist_train,
			loss_func = nn.CrossEntropyLoss(),
			optimizer=torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay=0.0001),
			epochs=5)
model.eval()
score = testing(model=model, dataset=mnist_test)
print (np.round(score*10000)/100)
path = os.path.join('./VSCODE PYTHON/Weights/Detect Num.pth')
torch.save(model.cpu().state_dict(), path)"""
############################################################



######################## VALIDATION ########################
"""for item in tqdm(hyperparametres):
	scores = []
	for train, test in tqdm(valid.split(X_TRAIN), total= val_of_batch, leave = False):
		x_train_1, x_test_1 = X_TRAIN[train], X_TRAIN[test]
		y_train_1, y_test_1 = Y_TRAIN[train], Y_TRAIN[test]

		TRAIN_DATA = torch.utils.data.TensorDataset(x_train_1, y_train_1)
		VALID_DATA = torch.utils.data.TensorDataset(x_test_1, y_test_1)

		model = neuralnet(hidden_dim= item['hidden_neur'], p_dropout=item['p'], num_layers = item['val_layers'], softmax = True)
		model.train()
		training(model=model, dataset=TRAIN_DATA,
				 loss_func = nn.CrossEntropyLoss(),
				 optimizer=torch.optim.Adam(model.parameters(), lr= item['lr'], weight_decay=item['l2']),
				 epochs=item['epochs'])
		model.eval()
		mean = testing(model=model, dataset=VALID_DATA)
		scores.append(mean)
	model_scores.append(np.mean(scores))
	print("model with: lr: {item[lr]}, 
                       val_layers: {item[val_layers]}, 
                       hidden_neur: {item[hidden_neur]}, 
                       p: {item[p]}, l2: {item[l2]}, 
                       epochs: {item[epochs]} 
                       HAS SCORE - {score}".format(item = item, score = model_scores[-1]))"""
############################################################



########### Загрузка и тестирование на реальном изображении ###########
model = neuralnet(hidden_dim=64, p_dropout=0.3, num_layers=1, softmax=True)
model.load_state_dict(torch.load('./VSCODE PYTHON/Weights/Detect Num.pth'))
im = Image.open("./VSCODE PYTHON/Neuron/Detect Num/1.jpg")

data_image = np.array(im, dtype=np.uint8)
data_image = data_image[:,:,0]
data_image = np.squeeze(data_image)
data_image1 = torch.from_numpy(data_image).float()
data_image1 = data_image1 / 255.0

print(predictt(model, data_image1))
#######################################################################