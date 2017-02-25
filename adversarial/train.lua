require 'torch'   
require 'xlua'    
require 'optim'  
require 'gnuplot'   
require 'io'   
require 'image'

----------------------------------------------------------------------

parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

----------------------------------------------------------------------

optimStateD = {
  learningRate = 0.1,
  weightDecay = 0.0,
  momentum = 0.5
}
optimStateG = {
  learningRate = 0.1,
  weightDecay = 0.0,
  momentum = 0.5
}
optimMethodD = optim.sgd
optimMethodG = optim.sgd

batchSize = 100

imgO = torch.CudaTensor(batchSize, 
                        trainData.data:size(2), 
                        trainData.data:size(3), 
                        trainData.data:size(4))
imgG = torch.CudaTensor(batchSize, 
                        trainData.data:size(2), 
                        trainData.data:size(3), 
                        trainData.data:size(4))                        
labelO = torch.ones(batchSize, 1):cuda()
labelG = torch.zeros(batchSize, 1):cuda()
z = torch.CudaTensor(batchSize, zSize)

lossDA = torch.zeros(maxEpoch)
lossGA = torch.zeros(maxEpoch)
logFile = io.open('log.csv', 'w')
logFile:write('train error, test error\n')

timer = torch.Timer()

lossD = 0
cntD = 0
lossG = 0
cntG = 0

----------------------------------------------------------------------

local function doBatch(mod, crit, data, labels)
  local output = mod:forward(data)
  local f = crit:forward(output, labels)
  local df_do = crit:backward(output, labels)
  mod:backward(data, df_do)
  return f
end

----------------------------------------------------------------------

local fevalDx = function(x)
  gradParametersD:zero()
  
  local f1 = doBatch(netD, criterion, imgO, labelO)
  lossD = lossD + f1
  cntD = cntD + 1
  
  netG:forward(z)
  imgG:copy(netG.output)
  local f2 = doBatch(netD, criterion, imgG, labelG)
  lossD = lossD + f2
  cntD = cntD + 1
  
  return f1 + f2, gradParametersD
end

----------------------------------------------------------------------

local fevalGx = function(x)
  gradParametersG:zero()
  local output = netD.output
  local f = criterion:forward(output, labelO)
  local df_do = criterion:backward(output, labelO)
  local df_dg = netD:updateGradInput(imgG, df_do)
  netG:backward(z, df_dg)
  lossG = lossG + f
  cntG = cntG + 1
  return f, gradParametersG
end

----------------------------------------------------------------------

function train()
  local shuffle = torch.randperm(trsize)
  lossD = 0
  cntD = 0
  lossG = 0
  cntG = 0
  --run batches
  for bnum = 1, trsize / batchSize do
    xlua.progress(bnum, trsize / batchSize)
    --get the img batch
    for i = 1, batchSize do
      local j = shuffle[i + (bnum - 1) * batchSize]
      imgO[i]:copy(trainData.data[j])
    end
    z:normal(0, 1)
    --do the discr batch
    optimMethodD(fevalDx, parametersD, optimStateD)
    --do the adver batch
    optimMethodG(fevalGx, parametersG, optimStateG)
  end
  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil
  parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = netG:getParameters()
  return lossD / cntD, lossG / cntG
end

----------------------------------------------------------------------

local function display()
  z:normal(0, 1)
  local output = netG:forward(z)
  local imgH = output:size(3)
  local imgW = output:size(4)
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
  lossD = lossD / cntD
  lossG = lossG / cntG
  -- Print errors and time
  print("Discriminator loss: " .. string.format("%10.5f", lossD))
  print("Generator loss:     " .. string.format("%10.5f", lossG)) 
  print("Time:               " .. string.format("%10.5f", timeTr))
  
  -- Write errors to file      
  logFile:write(string.format("%10.5f, %10.5f\n", lossD, lossG))
  logFile:flush()
  -- Plot errors
  if plotEnable then
    lossDA[epoch] = lossD
    lossGA[epoch] = lossG  
    gnuplot.plot({'Discriminator', lossDA:narrow(1, 1, epoch), '-'},
                 {'Generator', lossGA:narrow(1, 1, epoch), '-'})   
  end
  -- Save model             
  if epoch % modelSaveEpochs == 0 then
    saveModel(epoch)             
  end
  -- Reset loss counters
  lossD = 0
  cntD = 0
  lossG = 0
  cntG = 0
  -- Display
  display()
end

logFile:close()

