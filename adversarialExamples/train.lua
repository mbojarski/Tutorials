require 'torch'   
require 'xlua'    
require 'optim'  
require 'gnuplot'   
require 'io'
require 'image'

----------------------------------------------------------------------

classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion1 = optim.ConfusionMatrix(classes)
confusion2 = optim.ConfusionMatrix(classes)

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

adverGain = 5

data = torch.CudaTensor(batchSize, 
                        trainData.data:size(2), 
                        trainData.data:size(3), 
                        trainData.data:size(4))
noise = torch.CudaTensor(batchSize,
                        trainData.data:size(2),
                        trainData.data:size(3),
                        trainData.data:size(4))
adver = torch.CudaTensor(batchSize,
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

local function genAdver(mod, crit, data, labels)
  local output = mod:forward(data)
  crit:forward(output, labels)
  local df_do = crit:backward(output, labels)
  df_do:mul(-1):add(-0.01)
  noise:copy(mod:backward(data, df_do))
  noise:mul(-adverGain)
  adver:copy(data):add(noise):clamp(0, 1)  
end

----------------------------------------------------------------------

local feval = function(x)
  gradParameters:zero()
  genAdver(model, criterion, data, labels)
  gradParameters:zero()
  local f1 = doBatch(model, criterion, data, labels)
  confusion1:batchAdd(model.output, labels)
  local f2 = doBatch(model, criterion, adver, labels)
  confusion2:batchAdd(model.output, labels)
  return f1 + f2, gradParameters
end

----------------------------------------------------------------------

local function display()
  local out1 = adver 
  local out2 = data
  local imgH = adver:size(3)
  local imgW = adver:size(4)
  local img = torch.ones(imgH * 10 + 9, imgW * 20 + 19)
  for x = 1, 10 do
    for y = 1, 10 do
      img:narrow(1, (y - 1) * (imgH + 1) + 1, imgH):
          narrow(2, (x - 1) * (imgW + 1) + 1, imgW):
          copy(out1[(y - 1) * 10 + x][1]:float())
      img:narrow(1, (y - 1) * (imgH + 1) + 1, imgH):
          narrow(2, (x + 9) * (imgW + 1) + 1, imgW):
          copy(out2[(y - 1) * 10 + x][1]:float())
    end
  end
  img:clamp(0 ,1)
  disp1 = image.display{image=img, win=disp1, zoom=2}
end

----------------------------------------------------------------------

local function train()
  model:training()
  confusion1:zero()
  confusion2:zero()
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
  confusion1:updateValids()
  confusion2:updateValids()
  local errTr1 = 1 - confusion1.averageValid
  local errTr2 = 1 - confusion2.averageValid
  return errTr1 * 100, errTr2 * 100
end

----------------------------------------------------------------------

local function test()
  model:evaluate()
  confusion1:zero()
  --run batches
  for bnum = 1, tesize / batchSize do
    xlua.progress(bnum, tesize / batchSize)
    --create batch
    data:copy(testData.data:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    labels:copy(testData.labels:narrow(1, (bnum - 1) * batchSize + 1, batchSize))
    --test with batch   
    local output = model:forward(data)
    local f = criterion:forward(output, labels)
    confusion1:batchAdd(model.output, labels) 
  end
  confusion1:updateValids()
  local errTe = 1 - confusion1.averageValid
  return errTe * 100
end

----------------------------------------------------------------------

for epoch = 1, maxEpoch do
  print('Epoch ' .. epoch)
  -- Train
  timer:reset()
  local errTr1, errTr2 = train()
  local timeTr = timer:time().real  
  display()
  -- Test
  timer:reset()
  local errTe = test()
  local timeTe = timer:time().real
  -- Print errors and time
  print("Train error: " .. string.format("%10.5f", errTr1) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTr))
  print("Adver error: " .. string.format("%10.5f", errTr2) ..
        "\t\ttime: "  .. string.format("%10.5f", timeTr))
  print("Test error:  " .. string.format("%10.5f", errTe) .. 
        "\t\ttime: "  .. string.format("%10.5f", timeTe)) 
  -- Write errors to file      
  logFile:write(string.format("%10.5f, %10.5f\n", errTr1, errTe))
  logFile:flush()
  -- Plot errors
  if plotEnable then
    errTrA[epoch] = errTr1
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

