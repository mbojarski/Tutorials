require 'torch'  
require 'nn'      
require 'cunn'

----------------------------------------------------------------------

noutputs = 10
local modelSelect = 3

if modelSelect == 1 then
  model = nn.Sequential()
  model:add(nn.SpatialBatchNormalization(1)) 
  model:add(nn.SpatialConvolution(1, 4, 5, 5, 2, 2)) 
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(4, 6, 5, 5, 2, 2)) 
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(6, 32, 5, 5))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(32, noutputs, 1, 1))
  model:add(nn.View(noutputs))  
  model:add(nn.LogSoftMax())
elseif modelSelect == 2 then
  model = nn.Sequential()
  model:add(nn.SpatialBatchNormalization(1)) 
  model:add(nn.Reshape(1024))
  model:add(nn.Linear(1024, 128)) 
  model:add(nn.ReLU(true))
  model:add(nn.Linear(128, 10))   
  model:add(nn.LogSoftMax())
elseif modelSelect == 3 then
  model = nn.Sequential()
  model:add(nn.SpatialBatchNormalization(1)) 
  model:add(nn.SpatialConvolution(1, 4, 5, 5)) 
  model:add(nn.ReLU(true))
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolution(4, 6, 5, 5)) 
  model:add(nn.ReLU(true))
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Reshape(150)) 
  model:add(nn.Dropout())
  model:add(nn.Linear(150, 256))
  model:add(nn.ReLU(true))
  model:add(nn.Dropout()) 
  model:add(nn.Linear(256, 10))  
  model:add(nn.LogSoftMax())
end

model:cuda()

----------------------------------------------------------------------

criterion = nn.ClassNLLCriterion()
criterion:cuda()

----------------------------------------------------------------------

print 'loss function:'
print(criterion)

----------------------------------------------------------------------

print 'model:'
print(model)

----------------------------------------------------------------------

