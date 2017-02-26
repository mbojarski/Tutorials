----------------------------------------------------------------------

maxEpoch = 100

modelFile = 'model'
modelSaveEpochs = maxEpoch

plotEnable = false

----------------------------------------------------------------------

dofile 'generateData.lua'
dofile 'model.lua'
dofile 'saveModel.lua'
dofile 'train.lua'


