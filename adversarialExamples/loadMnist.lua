require 'cutorch'

----------------------------------------------------------------------

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

trainFile = 'mnist.t7/train_32x32.t7'
testFile = 'mnist.t7/test_32x32.t7'

----------------------------------------------------------------------

print '==> loading dataset'

trainData = nil
testData = nil
trainData = torch.load(trainFile,'ascii')
testData = torch.load(testFile,'ascii')

trainData.data = trainData.data:cuda():div(255)
trainData.labels = trainData.labels:cuda()
testData.data = testData.data:cuda():div(255)
testData.labels = testData.labels:cuda()

trsize = trainData.data:size(1) / 10
tesize = testData.data:size(1)

----------------------------------------------------------------------

