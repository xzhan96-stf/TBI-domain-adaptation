import numpy as np
#from Code import KMM
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)

class LinearGeneratorA(nn.Module):

    def __init__(self, input_dimA, output_dim, dim):
        super(LinearGeneratorA, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimA, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(dim),
            nn.Linear(dim, output_dim)) # Real-value range

    def forward(self, x):
        return self.layers(x)


class LinearGeneratorB(nn.Module):

    def __init__(self, input_dimB, output_dim, dim):
        super(LinearGeneratorB, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimB, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(dim),
            nn.Linear(dim, output_dim)) # Real-value range

    def forward(self, x):
        return self.layers(x)

class BiGANDiscriminatorA(nn.Module):
    def __init__(self, latent_dim, dim):
        super(BiGANDiscriminatorA, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class BiGANDiscriminatorB(nn.Module):
    def __init__(self, latent_dim, dim):
        super(BiGANDiscriminatorB, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()) # To probability

    def forward(self, x):  # The noise has to be implicitly added into the generator
        return self.layers(x)

# class BiGANDiscriminatorA(nn.Module):
#     def __init__(self, latent_dim, dim):
#         super(BiGANDiscriminatorA, self).__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(latent_dim * 2, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, 1),
#             nn.Sigmoid())
#
#     def forward(self, x, z):
#         xz = torch.cat((x, z), dim=1) # The noise has to be implicitly added into the generator
#         return self.layers(xz)
#
#
# class BiGANDiscriminatorB(nn.Module):
#     def __init__(self, latent_dim, dim):
#         super(BiGANDiscriminatorB, self).__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(latent_dim * 2, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, 1),
#             nn.Sigmoid()) # To probability
#
#     def forward(self, x, z):  # The noise has to be implicitly added into the generator
#         xz = torch.cat((x, z), dim=1)
#         return self.layers(xz)

input_dimA = 510
input_dimB = 510
output_dimA = 510
output_dimB = 510
dim = 300 #Hidden layer neurons in the generator and discriminator

epochs = 1000
lr = 5e-3
lr2 = 1e-4
lr3 = 1e-3
beta = 1 #Balance the reconstruction loss and discriminator loss (fake)
alpha = 1 #Balance the generator loss and discriminator loss
EPS = 1e-6
momentum = 0.9
batch_size = 150
noise =0.2

# ------- Define Model ------- #

encoderA= LinearGeneratorA(input_dimA,output_dimB, dim) #From A to B
discriminatorA = BiGANDiscriminatorA(output_dimA, dim)

encoderB= LinearGeneratorB(input_dimB,output_dimA, dim) # From B to A
discriminatorB = BiGANDiscriminatorB(output_dimB, dim)

# ------- Optimizer ------- #

#opt_g = optim.Adam(list(encoderA.parameters()), lr= lr)
opt_g = optim.Adam(list(encoderA.parameters()) + list(encoderB.parameters()), lr= lr2) # Question: different learning rate?
opt_d = optim.Adam(list(discriminatorA.parameters())+list(discriminatorB.parameters()), lr= lr)
opt_e = optim.Adam(list(encoderA.parameters()) + list(encoderB.parameters()),lr= lr) #Question: 1 or 2 sets of generator parameters

loss = nn.BCELoss()

# ------- Directory ------ #
# This is the location of the data from Xianghao Zhan's folder (user may need to change)
Data_X_dir = 'G:/我的云端硬盘/Paper/GAN/Data/X'
Data_MPS_dir = 'G:/我的云端硬盘/Paper/GAN/Data/Y/MPS'
Data_MPSR_dir = 'G:/我的云端硬盘/Paper/GAN/Data/Y/MPSR'
Data_generated_dir = 'G:/我的云端硬盘/Paper/GAN/Data/GEN'
Dir_models = 'G:/我的云端硬盘/Paper/GAN/GAN models'


# ------- Load Data ------ #
# Load X data
from scipy import io
os.chdir(Data_X_dir)
HMXYZ_X = io.loadmat('HMXYZ_X.mat')['MLHM2_X']
HMXNYZ_X = io.loadmat('HMXNYZ_X.mat')['MLHM2_X']
HMXZY_X = io.loadmat('HMXZY_X.mat')['MLHM2_X']
HMXZNY_X = io.loadmat('HMXZNY_X.mat')['MLHM2_X']
HMYXZ_X = io.loadmat('HMYXZ_X.mat')['MLHM2_X']
HMYZX_X = io.loadmat('HMYZX_X.mat')['MLHM2_X']
HMZXY_X = io.loadmat('HMZXY_X.mat')['MLHM2_X']
HMNYZX_X = io.loadmat('HMNYZX_X.mat')['MLHM2_X']
HMZYX_X = io.loadmat('HMZYX_X.mat')['MLHM2_X']
HMZXNY_X = io.loadmat('HMZXNY_X.mat')['MLHM2_X']
HMZNYX_X = io.loadmat('HMZNYX_X.mat')['MLHM2_X']
HMNYXZ_X = io.loadmat('HMNYXZ_X.mat')['MLHM2_X']
HM_X = np.row_stack((HMXYZ_X, HMXNYZ_X, HMXZY_X, HMXZNY_X, HMYXZ_X, HMYZX_X, HMZXY_X, HMNYZX_X, HMZYX_X, HMZXNY_X,
                      HMZNYX_X, HMNYXZ_X))

xsacler = StandardScaler()
HM_X = xsacler.fit_transform(HM_X)

MMA1_X = io.loadmat('MMA79_X.mat')['MLHM2_X']
MMA2_X= io.loadmat('MMA_Tiernan_X.mat')['MLHM2_X']
MMA_X = np.row_stack([MMA1_X,MMA2_X])
assert MMA_X.shape[0] == 457

MMA_X = xsacler.transform(MMA_X)

#X and Y, need to change to our data
XA = HM_X
XB = MMA_X

XA_train = XA
XB_train = XB
XA_test = XA
XB_test = XB

# ------- Variables ------- #

scr_train = Variable(torch.from_numpy(XA_train).float())
tgt_train = Variable(torch.from_numpy(XB_train).float())
scr_test = Variable(torch.from_numpy(XA_test).float())
tgt_test = Variable(torch.from_numpy(XB_test).float())

s_train = torch.tensor(scr_train)
s_train_tensor = torch.utils.data.TensorDataset(s_train)
source_trainloader = torch.utils.data.DataLoader(dataset=s_train_tensor, batch_size=batch_size, shuffle=False)

t_train = torch.tensor(tgt_train)
t_train_tensor = torch.utils.data.TensorDataset(t_train)
target_trainloader = torch.utils.data.DataLoader(dataset=t_train_tensor, batch_size=batch_size, shuffle=False)

s_test = torch.tensor(scr_test)
s_test_tensor = torch.utils.data.TensorDataset(s_test)
source_testloader = torch.utils.data.DataLoader(dataset=s_test_tensor, batch_size=batch_size, shuffle=False)

t_test = torch.tensor(tgt_test)
t_test_tensor = torch.utils.data.TensorDataset(t_test)
target_testloader = torch.utils.data.DataLoader(dataset=t_test_tensor, batch_size=batch_size, shuffle=False)

loss_recorder = []
gloss_recorder = []
dloss_recorder = []
glossrec_recorder = []

true_acc_A_mean = []
fake_acc_A_mean = []
true_acc_B_mean = []
fake_acc_B_mean = []

pbar = tqdm.tqdm(range(epochs))
for e in pbar:

    # No need to perform train-test partition
    # XA_train, XA_test, yA_train, yA_test = train_test_split(XA, yA, test_size=0.2, random_state= 0)
    # XB_train, XB_test, yB_train, yB_test = train_test_split(XB, yB, test_size=0.2, random_state= 0)

    true_acc = []
    fake_acc = []
    true_acc_B = []
    fake_acc_B = []
    for (datas, datat) in zip(source_trainloader, target_trainloader):
        src_data = datas[0]
        tgt_data = datat[0]

        #For the Wasserstein GAN
        if src_data.size()[0] != batch_size:
            continue

        if tgt_data.size()[0] != batch_size:
            continue

        encoderA.train()
        encoderB.train()
        discriminatorA.train()
        discriminatorB.train()
        opt_d.zero_grad()
        opt_g.zero_grad()
        opt_e.zero_grad()

        validA_target = torch.ones((src_data.size()[0], 1))
        fakeA_target = torch.zeros((tgt_data.size()[0], 1)) # The fake samples converted from the target domain

        validB_target = torch.ones((tgt_data.size()[0], 1))
        fakeB_target = torch.zeros((src_data.size()[0], 1)) # The fake samples converted from the source domain

        src_gen_A = encoderA(src_data*torch.from_numpy(np.random.binomial(size=src_data.size(), n=1, p=1-noise)))# From A to B (HM->MMA)
        tgt_gen_B = encoderB(tgt_data*torch.from_numpy(np.random.binomial(size=tgt_data.size(), n=1, p=1-noise))) # From B to A (MMA->HM)

        tgt_gen_BA = encoderA(tgt_gen_B*torch.from_numpy(np.random.binomial(size=tgt_data.size(), n=1, p=1-noise)))  # Recovery: target->source->target (MMA->HM->MMA)
        src_gen_AB = encoderB(src_gen_A*torch.from_numpy(np.random.binomial(size=src_data.size(), n=1, p=1-noise)))  # Recovery: source->target->source (HM->MMA->HM)

        # loss_gA = torch.mean(torch.square(src_gen_A - tgt_gen_BA)) #Reconstruction loss
        # loss_gB = torch.mean(torch.square(tgt_gen_B - src_gen_AB)) #Reconstruction loss

        loss_gA_rec = torch.mean(torch.square(src_data - src_gen_AB))  # Reconstruction loss
        loss_gB_rec = torch.mean(torch.square(tgt_data - tgt_gen_BA))  # Reconstruction loss

        ### The following Four Sentences have been changed ###
        # discriminator_loss_real_A = discriminatorA(src_data, tgt_gen_B)
        # discriminator_loss_fake_A = discriminatorA(tgt_gen_B, src_gen_AB) #Here is weird.
        #
        # discriminator_loss_real_B = discriminatorB(tgt_data, src_gen_A)
        # discriminator_loss_fake_B = discriminatorB(src_gen_A, tgt_gen_BA) #Here is weird.

        discriminator_loss_real_A = discriminatorA(src_data)
        discriminator_loss_fake_A = discriminatorA(tgt_gen_B)

        discriminator_loss_real_B = discriminatorB(tgt_data)
        discriminator_loss_fake_B = discriminatorB(src_gen_A)

        true_acc.append(np.mean(discriminator_loss_real_A.detach().numpy()>0.5))
        fake_acc.append(np.mean(discriminator_loss_fake_A.detach().numpy()<0.5))
        true_acc_B.append(np.mean(discriminator_loss_real_B.detach().numpy() > 0.5))
        fake_acc_B.append(np.mean(discriminator_loss_fake_B.detach().numpy() < 0.5))

        # loss_dA = loss(discriminator_loss_real_A, validA_target) + loss(discriminator_loss_fake_A, fakeA_target)
        # loss_dB = loss(discriminator_loss_real_B, validB_target) + loss(discriminator_loss_fake_B, fakeB_target)
        # For the nonsaturating loss
        loss_dA = -torch.mean(torch.log(discriminator_loss_real_A+EPS)) - torch.mean(torch.log(1-discriminator_loss_fake_A+EPS))
        loss_dB = -torch.mean(torch.log(discriminator_loss_real_B+EPS)) - torch.mean(torch.log(1-discriminator_loss_fake_B+EPS))

        #lossG = (loss_gA_rec + loss_gB_rec)/alpha - beta*(loss(discriminator_loss_fake_A, fakeA_target) + loss(discriminator_loss_fake_B, fakeB_target))
        #lossGfake = -(loss(discriminator_loss_fake_A, fakeA_target) + loss(discriminator_loss_fake_B, fakeB_target))
        lossGfake = -(torch.mean(torch.log(discriminator_loss_fake_A+EPS))+torch.mean(torch.log(discriminator_loss_fake_B+EPS)))
        lossD = loss_dA + loss_dB
        lossGrec = (loss_gA_rec + loss_gB_rec)
        total_loss = lossGrec/alpha + beta*lossGfake + lossD

        #total_loss.backward()
        lossD.backward(retain_graph=True)
        lossGrec.backward(retain_graph=True)
        lossGfake.backward()
        opt_d.step()
        opt_g.step()
        opt_e.step()
    loss_recorder.append(total_loss)
    gloss_recorder.append(lossGfake)
    glossrec_recorder.append(lossGrec)
    dloss_recorder.append(lossD)
    true_acc_A_mean.append(np.mean(true_acc))
    fake_acc_A_mean.append(np.mean(fake_acc))
    true_acc_B_mean.append(np.mean(true_acc_B))
    fake_acc_B_mean.append(np.mean(fake_acc_B))
    pbar.set_description('Bi-GAN Total Loss: %.2e, G-Rec Loss: %.2e, G-Fake Loss: %.2e, D Loss: %.2e, A True ACC: %.2f, A Fake ACC: %.2f , B True ACC: %.2f, B Fake ACC: %.2f' % (total_loss.item(), lossGrec.item(), lossGfake.item(), lossD.item(), np.mean(true_acc), np.mean(fake_acc), np.mean(true_acc_B), np.mean(fake_acc_B)))

# Save models
os.chdir(Dir_models)
specs = 'Noisy'+ str(noise)+'_Bi-GAN_HM2MMA_standardization_five layers_nonsaturating_separateGDE_dim' + str(dim) +'_epoch'+str(epochs) + '_G initial_lr' + str(lr) + '_D initial_lr' + str(lr3)+ '_Grec initial_lr' + str(lr2)+ '_beta' + str(beta) + '_alpha' + str(alpha)
torch.save({
    'epoch': epochs,
    'encoderA_state_dict': encoderA.state_dict(),
    'encoderB_state_dict': encoderB.state_dict(),
    'discriminatorA_state_dict': discriminatorA.state_dict(),
    'discriminatorB_state_dict': discriminatorB.state_dict(),
    'opt_g_state_dict': opt_g.state_dict(),
    'opt_d_state_dict': opt_d.state_dict(),
    'opt_e_state_dict': opt_e.state_dict(),
    'loss': loss_recorder[-1]}, specs + '.pt')
print('Model saved to disk!')


os.chdir(Dir_models)
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(loss_recorder,'r')
plt.plot(glossrec_recorder,'c:',linewidth=3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(['Total Loss','Generator Faking Loss','Discriminator Loss','Generator Reconstruction Loss'],fontsize=18)
plt.title('Total Loss and Reconstruction Loss')
plt.savefig(specs+'loss.pdf',bbox_inches='tight')
plt.show()
plt.figure(figsize=(9,6))
plt.plot(gloss_recorder,'b')
plt.plot(dloss_recorder,'g:',linewidth=3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(['Generator Faking Loss','Discriminator Loss'],fontsize=18)
plt.title('GD Loss')
plt.savefig(specs+'gdloss.pdf',bbox_inches='tight')
plt.show()
plt.figure(figsize=(9,6))
plt.plot(true_acc_A_mean,'r:',linewidth=3)
plt.plot(fake_acc_A_mean,'b')
plt.plot(true_acc_B_mean,'g:',linewidth=3)
plt.plot(fake_acc_B_mean,'c')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.legend(['Real HM Classification Accuracy','Fake HM Classification Accuracy','Real MMA Classification Accuracy','Fake MMA Classification Accuracy'],fontsize=18)
plt.title('Classification Accuracy')
plt.savefig(specs+'accuracy.pdf',bbox_inches='tight')
plt.show()

for datas in target_testloader:
    tgt_data = datas[0]
    tgt_gen_B = encoderB(tgt_data).detach().numpy()
    os.chdir(Data_generated_dir)
    np.save(specs+'.npy',tgt_gen_B)
