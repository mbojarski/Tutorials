require 'torch'  
require 'nn'      
require 'cunn'

----------------------------------------------------------------------

model = nn.Sequential()
model:add(nn.Linear(2, 256)) 
model:add(nn.ReLU(true))
model:add(nn.Linear(256, 1))  
model:float()

----------------------------------------------------------------------

criterion = nn.MSECriterion()
criterion:float()

----------------------------------------------------------------------

print 'loss function:'
print(criterion)

----------------------------------------------------------------------

print 'model:'
print(model)

----------------------------------------------------------------------

