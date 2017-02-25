require 'torch'  
require 'nn'      
require 'cunn'

----------------------------------------------------------------------

local function setWeights(module, std)
  weight = module.weight
  if weight then
    weight:randn(weight:size()):mul(std)
  end
  bias = module.bias
  if bias then
    bias:zero()
  end
end

----------------------------------------------------------------------

local function initModel(model, std)
  for _, m in pairs(model:listModules()) do
    setWeights(m, std)
  end
end

----------------------------------------------------------------------

zSize = 100
hG = {256, 512}

netG = nn.Sequential()
netG:add(nn.Linear(zSize, hG[1])) 
netG:add(nn.ReLU(true))
netG:add(nn.Linear(hG[1], hG[2])) 
netG:add(nn.ReLU(true))
netG:add(nn.Linear(hG[2], 1024))  
netG:add(nn.Reshape(1, 32, 32)) 
netG:add(nn.Sigmoid())

netG:cuda()
initModel(netG, 0.05)

----------------------------------------------------------------------

hD = {512, 256}

netD = nn.Sequential() 
netD:add(nn.Reshape(1024))
netD:add(nn.Linear(1024, hD[1])) 
netD:add(nn.ReLU(true))
netD:add(nn.Dropout())
netD:add(nn.Linear(hD[1], hD[2])) 
netD:add(nn.ReLU(true))
netD:add(nn.Dropout())
netD:add(nn.Linear(hD[2], 1))  
netD:add(nn.Sigmoid())

netD:cuda()
initModel(netD, 0.005)

----------------------------------------------------------------------

criterion = nn.BCECriterion()
criterion:cuda()

----------------------------------------------------------------------

print 'loss function:'
print(criterion)

----------------------------------------------------------------------

print 'netG:'
print(netG)
print 'netD:'
print(netD)

----------------------------------------------------------------------

