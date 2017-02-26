require 'torch'   
require 'xlua'    
require 'optim'  
require 'gnuplot'   
require 'io'   
require 'image'

----------------------------------------------------------------------

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

data = torch.FloatTensor(batchSize, trainData.data:size(2))
labels = torch.FloatTensor(batchSize)

lossTrA = torch.zeros(maxEpoch)
lossTeA = torch.zeros(maxEpoch)
logFile = io.open('log.csv', 'w')
logFile:write('train error, test error\n')

timer = torch.Timer()

lossTr = 0
cntTr = 0
lossTe = 0
cntTe = 0

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
  lossTr = lossTr + f1
  cntTr = cntTr + 1
  return f1, gradParameters
end

----------------------------------------------------------------------

local function train()
  model:training()
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
end

----------------------------------------------------------------------

local function test()
  model:evaluate()
  --run batches
  for bnum = 1, tesize / batchSize do
    xlua.progress(bnum, tesize / batchSize)
    --create batch
    data:copy(testData.data:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    labels:copy(testData.labels:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    --test with batch   
    local output = model:forward(data)
    local f = criterion:forward(output, labels)
    lossTe = lossTe + f
    cntTe = cntTe + 1
  end
end

----------------------------------------------------------------------

for epoch = 1, maxEpoch do
  print('Epoch ' .. epoch)
  -- Train
  timer:reset()
  train()
  local timeTr = timer:time().real  
  lossTr = lossTr / cntTr
  -- Test
  timer:reset()
  test()
  local timeTe = timer:time().real
  lossTe = lossTe / cntTe
  -- Print errors and time
  print("Train error: " .. string.format("%10.5f", lossTr) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTr))
  print("Test error:  " .. string.format("%10.5f", lossTe) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTe)) 
  -- Write errors to file      
  logFile:write(string.format("%10.5f, %10.5f\n", lossTr, lossTe))
  logFile:flush()
  -- Plot errors
  if plotEnable then
    lossTrA[epoch] = lossTr
    lossTeA[epoch] = lossTe  
    gnuplot.plot({'train', lossTrA:narrow(1, 1, epoch), '-'},
                 {'test', lossTeA:narrow(1, 1, epoch), '-'})   
  end
  -- Save model             
  if epoch % modelSaveEpochs == 0 then
    saveModel(epoch)             
  end
  -- Reset loss counters
  lossTr = 0
  cntTr = 0
  lossTe = 0
  cntTe = 0
end

logFile:close()

