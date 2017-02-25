require 'torch'   
require 'xlua'    
require 'optim'  
require 'gnuplot'   
require 'io'

----------------------------------------------------------------------

classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)

if model then
   parameters, gradParameters = model:getParameters()
end

----------------------------------------------------------------------

optimState = {
  learningRate = 0.1,
  weightDecay = 0.0,
  momentum = 0.9
}
optimMethod = optim.sgd

batchSize = 100

data = torch.CudaTensor(batchSize, 
                        trainData.data:size(2), 
                        trainData.data:size(3), 
                        trainData.data:size(4))
labels = torch.CudaTensor(batchSize)

errTrA = torch.zeros(maxEpoch)
errTeA = torch.zeros(maxEpoch)
logFile = io.open('log.csv', 'w')
logFile:write('train error, test error\n')

timer = torch.Timer()

----------------------------------------------------------------------

local function doBatch(mod, crit, data, labels)
  local output = mod:forward(data)
  local f = crit:forward(output, labels)
  local df_do = crit:backward(output, labels)
  mod:backward(data, df_do)
  return f
end

----------------------------------------------------------------------

local feval = function(x)
  gradParameters:zero()
  local f1 = doBatch(model, criterion, data, labels)
  confusion:batchAdd(model.output, labels)
  return f1, gradParameters
end

----------------------------------------------------------------------

local function train()
  model:training()
  confusion:zero()
  local shuffle = torch.randperm(trsize)
  --run batches
  for bnum = 1, trsize / batchSize do
    xlua.progress(bnum, trsize / batchSize)
    --create batch
    for i = 1, batchSize do
      local j = shuffle[i + (bnum - 1) * batchSize]
      data[i]:copy(trainData.data[j])
      labels[i] = trainData.labels[j]
    end   
    --train with batch 
    optimMethod(feval, parameters, optimState)    
  end
  confusion:updateValids()
  local errTr = 1 - confusion.averageValid
  return errTr
end

----------------------------------------------------------------------

local function test()
  model:evaluate()
  confusion:zero()
  --run batches
  for bnum = 1, tesize / batchSize do
    xlua.progress(bnum, tesize / batchSize)
    --create batch
    data:copy(testData.data:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    labels:copy(testData.labels:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    --test with batch   
    local output = model:forward(data)
    local f = criterion:forward(output, labels)
    confusion:batchAdd(model.output, labels) 
  end
  confusion:updateValids()
  local errTe = 1 - confusion.averageValid
  return errTe
end

----------------------------------------------------------------------

for epoch = 1, maxEpoch do
  print('Epoch ' .. epoch)
  -- Train
  timer:reset()
  local errTr = train() * 100
  local timeTr = timer:time().real  
  -- Test
  timer:reset()
  local errTe = test() * 100
  local timeTe = timer:time().real
  -- Print errors and time
  print("Train error: " .. string.format("%10.5f", errTr) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTr))
  print("Test error:  " .. string.format("%10.5f", errTe) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTe)) 
  -- Write errors to file      
  logFile:write(string.format("%10.5f, %10.5f\n", errTr, errTe))
  logFile:flush()
  -- Plot errors
  if plotEnable then
    errTrA[epoch] = errTr
    errTeA[epoch] = errTe  
    gnuplot.plot({'train', errTrA:narrow(1, 1, epoch), '-'},
                 {'test', errTeA:narrow(1, 1, epoch), '-'})   
  end
  -- Save model             
  if epoch % modelSaveEpochs == 0 then
    saveModel(epoch)             
  end
end

logFile:close()

