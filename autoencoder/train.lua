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

data = torch.CudaTensor(batchSize, 
                        trainData.data:size(2), 
                        trainData.data:size(3), 
                        trainData.data:size(4))

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
  local f1 = doBatch(model, criterion, data, data)
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
    --test with batch   
    local output = model:forward(data)
    local f = criterion:forward(output, data)
    lossTe = lossTe + f
    cntTe = cntTe + 1
  end
end

----------------------------------------------------------------------

local function display()
  model:evaluate()
  data:copy(testData.data:narrow(1, 1, 100))
  local output = model:forward(data)
  local imgH = testData.data:size(3)
  local imgW = testData.data:size(4)
  local img = torch.ones(imgH * 10 + 9, imgW * 10 + 9)
  for x = 1, 10 do
    for y = 1, 10 do
      img:narrow(1, (y - 1) * (imgH + 1) + 1, imgH):
          narrow(2, (x - 1) * (imgW + 1) + 1, imgW):
          copy(output[(y - 1) * 10 + x][1]:float())
    end
  end
  disp1 = image.display{image=img, win=disp1, zoom=2}
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
  -- Display
  display()
end

logFile:close()

