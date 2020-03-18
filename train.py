import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime

from evalution_segmentaion import eval_semantic_segmentation # 指标计算
import data_augmentation # 数据处理
from FCN import FCN # 模型载入

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')	# 指定训练方式

BATCH_SIZE = 2
train_data = DataLoader(data_augmentation.Cam_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_data = DataLoader(data_augmentation.Cam_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

net = FCN(12)
net = net.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

eval_miou_list = []
best = [0]
print('-----------------------train-----------------------')


for epoch in range(500):
	if epoch % 50 == 0 and epoch != 0:
		for group in optimizer.param_groups:
			group['lr'] *= 0.5

	train_loss = 0
	train_acc = 0
	train_miou = 0
	train_class_acc = 0

	net = net.train()
	prec_time = datetime.now()

	for i, sample in enumerate(train_data):
		imgdata = Variable(sample['img'].to(device))
		imglabel = Variable(sample['label'].long().to(device))

		optimizer.zero_grad()
		out = net(imgdata)
		out = F.log_softmax(out, dim=1)

		loss = criterion(out, imglabel)

		loss.backward()
		optimizer.step()
		train_loss = loss.item() + train_loss

		pre_label = out.max(dim=1)[1].data.cpu().numpy()
		pre_label = [i for i in pre_label]

		true_label = imglabel.data.cpu().numpy()
		true_label = [i for i in true_label]

		eval_metrix = eval_semantic_segmentation(pre_label, true_label)
		train_acc = eval_metrix['mean_class_accuracy'] + train_acc
		train_miou = eval_metrix['miou'] + train_miou
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']

	net = net.eval()
	eval_loss = 0
	eval_acc = 0
	eval_miou = 0
	eval_class_acc = 0

	for j, sample in enumerate(val_data):
		valImg = Variable(sample['img'].to(device))
		valLabel = Variable(sample['label'].long().to(device))

		out = net(valImg)
		out = F.log_softmax(out, dim=1)
		loss = criterion(out, valLabel)
		eval_loss = loss.item() + eval_loss
		pre_label = out.max(dim=1)[1].data.cpu().numpy()
		pre_label = [i for i in pre_label]

		true_label = valLabel.data.cpu().numpy()
		true_label = [i for i in true_label]

		eval_metrics = eval_semantic_segmentation(pre_label, true_label)
		eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
		eval_miou = eval_metrics['miou'] + eval_miou
		eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

	cur_time = datetime.now()
	h, remainder = divmod((cur_time - prec_time).seconds, 3600)
	m, s = divmod(remainder, 60)

	epoch_str = ('|Epoch|: {}\n|Train Loss|: {:.5f}\n|Train Acc|: {:.5f}\n|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}\n'
				 '|Valid Loss|: {:.5f}\n|Valid Acc|: {:.5f}\n|Valid Mean IU|: {:.5f}\n|Valid Class Acc|:{:}'.format(
				  epoch, train_loss / len(train_data), train_acc / len(train_data), train_miou / len(train_data),
				  train_class_acc / len(train_data), eval_loss / len(train_data), eval_acc/len(val_data),
				  eval_miou/len(val_data), eval_class_acc / len(val_data)))

	time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
	print(epoch_str + "\n" + time_str)

	if (max(best) <= eval_miou/len(val_data)):
		best.append(eval_miou/len(val_data))
		t.save(net.state_dict(),  'xxx.pth')

