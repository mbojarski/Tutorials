require 'torch'

----------------------------------------------------------------------

print '==> generating dataset'

trsize = 5000
tesize = 1000

trainData = {}
testData = {}

local xRange = {-1, 1}
local yRange = {-1, 1}

trainData.data = torch.FloatTensor(trsize, 2)
trainData.labels = torch.FloatTensor(trsize)
testData.data = torch.FloatTensor(tesize, 2)
testData.labels = torch.FloatTensor(tesize)

for i = 1, trsize do
    trainData.data[i][1] = torch.uniform(xRange[1], xRange[2])
    trainData.data[i][2] = torch.uniform(yRange[1], yRange[2])
    trainData.labels[i] = trainData.data[i][1]^2 + trainData.data[i][2]^2
end

for i = 1, tesize do
    testData.data[i][1] = torch.uniform(xRange[1], xRange[2])
    testData.data[i][2] = torch.uniform(yRange[1], yRange[2])
    testData.labels[i] = testData.data[i][1]^2 + testData.data[i][2]^2
end

----------------------------------------------------------------------

