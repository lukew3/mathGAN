import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

mydict = {'1': 1.,
          '2': 2.,
          '3': 3.,
          '4': 4.,
          '5': 5.,
          '6': 6.,
          '7': 7.,
          '8': 8.,
          '9': 9.,
          '0': 10.,
          '+': 11.,
          '-': 12.,
          '*': 13.,
          '/': 14.,
          '=': 15.
         }

mydict_inverted = {1.: "1",
                   2.: "2",
                   3.: "3",
                   4.: "4",
                   5.: "5",
                   6.: "6",
                   7.: "7",
                   8.: "8",
                   9.: "9",
                   10.: "0",
                   11.: "+",
                   12.: "-",
                   13.: "*",
                   14.: "/",
                   15.: "=",
                   0.: " "
                  }

def problem_to_tensor(problem_string):
    outlist = []
    for char in problem_string:
        outlist.append(mydict[char])
    for _ in range(0, 20-len(problem_string)):
        outlist.append(0)
    t = torch.as_tensor(outlist)
    return t

def tensor_to_problem(t):
    inlist = t.tolist()
    outstring = ""
    for myvalue in inlist[0]:
        #print(myvalue)
        #print(mydict_inverted[myvalue])
        outstring += mydict_inverted[myvalue] 
    outstring = outstring.strip()
    return outstring

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            # Takes 20 digits as in_features
            # Can the 20 be replaced with problem_dim?
            nn.Linear(in_features, 20),
            #Activation function
            nn.LeakyReLU(0.01),
            # One output node, 0 for fake, 1 for real
            nn.Linear(20, 1),
            # Makes sure that the node has a value between 0 and 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    # z_dim is the dimenion of the noise
    # problem_dim is the dimension of the output (length of string)
    def __init__(self, z_dim, problem_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # I think that 256 is just random to expand the noise
            # 256 = 64*4
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, problem_dim),
            # normalize inputs to [-1, 1] so make outputs [-1, 1]
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
#learning rate ( play around with this if you want)
lr = 3e-4
# noise dimension (play with this as well)
z_dim = 64 #try 128, 256
# Set dimension of problem, length of string
problem_dim = 20
batch_size = 1
num_epochs = 50

disc = Discriminator(problem_dim).to(device)
gen = Generator(z_dim, problem_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
# Params of Normalize are mean and standard deviation of dataset
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

class ProblemDataset(Dataset):
    def __init__(self, problem_list, transform=None):
        self.problem_list = problem_list
        self.transform = transform

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        problem_string = self.problem_list[index]
        problem_tensor = problem_to_tensor(problem_string)

        return problem_tensor

input_problems = ['32+90', '24+13', '93+03', '17+18', '68+03', '22+11', '50+50', '47+93', '08+29', '73+12']
# Not sure how transform ToTensor works, need an implementation later.
dataset = ProblemDataset(problem_list = input_problems)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(loader):
        # Keep current batch number, flatten real to size of 20
        real = real.view(-1, 20).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 20)
                # I think that data is the problem that the model was trained on
                data = real.reshape(-1, 20)
                #print(tensor_to_problem(fake))
                print(tensor_to_problem(data))
                
                # img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                # img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                #print(fake)
                # print(data)
                """
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                """
                step += 1