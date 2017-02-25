----------------------------------------------------------------------

maxEpoch = 100

modelFile = 'model'
modelSaveEpochs = maxEpoch

plotEnable = true

----------------------------------------------------------------------

dofile 'loadMnist.lua'
dofile 'model.lua'
dofile 'saveModel.lua'
dofile 'train.lua'


