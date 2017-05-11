--
-- Krishna Murthy, Sarthak Sharma
-- Januray 2017
--


-- Load requried packages
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'


----------------------
-- Helper Functions --
----------------------

-- Recover predictions from a bunch of heatmaps
-- Takes 'hm' - a set of heatmaps - as input
function getPreds(hm)

    -- We assume the 4 heatmap dimensions are for [num images] x [num kps per image] x [height] x [width]

    -- Verify that the heatmap tensor has 4 dimensions
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    -- Reshape the heatmap so that [height] and [width] are flattened out to a single dimension
    -- Get the maxima over the third dimension (comprising of the [height * width] flattened values)
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    -- Allocate memory for a tensor to hold the X, Y coordinates of the maxima, and the confidence score
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    -- Obtain the X coordinate of each maxima
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    -- Obtain the Y coordinate of each maxima
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    
    -- Return the predicted locations
    return preds

end


-- A function to perform slight post-processing, to be accurate to the pixel level
function postprocess(output, outputRes)
    
    -- Obtain keypoint predictions from the output heatmaps
    local p = getPreds(output)
    -- Initialize a tensor to hold the prediction confidences (pixel intensities at the output location)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1, p:size(1) do
        for j = 1, p:size(2) do
            local hm = output[i][j]
            local pX, pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < outputRes and pY > 1 and pY < outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)
   
    return p:cat(p,3):cat(scores,3)

end


-----------------
-- Main Script --
-----------------

-- Set the default tensor type to a FloatTensor
torch.setdefaulttensortype('torch.FloatTensor')

-- Number of keypoints (set according to the object category)
numKps = 14

-- Number of hourglass modules stacked
numStack = 2


-- Dimensions of each prediction
predDim = {numKps, 5}
-- Dimension of each input to the network
inputDim = {3, 64, 64}
-- Dimension of each output from the newtork
outputDim = {}
for i = 1,numStack do
    outputDim[i] = {numKps, 64, 64}
end
-- Resolution of the output image (assumed to be square)
outputRes = 64


-- Input file (contains image paths and bboxes)
-- Syntax of each line: /full/path/to/image x y w h
-- Here, the image refers to the entire image (eg. a KITTI frame)
-- x, y, w, h are "0-based" indices of a car bounding box
dataPath = '/home/km/code/hourglass-test/exp/testInstances.txt'

-- Path to the saved model (.t7 file)
modelPath = '/home/km/code/stacked-hourglass/models/singleGPUModel.t7'

-- Path to the results file, where keypoint predictions will be written
resultPath = '/home/km/code/stacked-hourglass/exp/results.txt'

-- Determine the number of images
local numImgs = 0;
for line in io.lines(dataPath) do 
    numImgs = numImgs + 1;
end

-- Initialize variables to save the images, predictions, and their heatmaps
saved = {idxs = torch.Tensor(numImgs), preds = torch.Tensor(numImgs, unpack(predDim))}
-- saved.input = torch.Tensor(numImgs, unpack(inputDim))
-- saved.heatmaps = torch.Tensor(numImgs, unpack(outputDim[1]))

-- Load the model
print('Loading the model ...')
model = torch.load(modelPath)
model:cuda()

print('Predicting keypoints')

-- For each instance whose kps are to be predicted
i = 1;
for line in io.lines(dataPath) do
    
    -- Load the image from a text file,format : /path/to/text/file x y w h
    cimgpath, cx, cy, cw, ch = unpack(line:split(" "));
    -- Image path
    cimg = image.load(cimgpath)
    -- (0-based) X coordinate of top left corner of bbox
    cx = tonumber(cx);
    -- (0-based) Y coordinate of top left corner of bbox
    cy = tonumber(cy);
    -- Width of bbox
    cw = tonumber(cw);
    -- Height of bbox
    ch = tonumber(ch);
   
    -- Converting the image to a float tensor (by default, images are loaded as userdata, not tensors)
    cimg = torch.FloatTensor(cimg);
    -- Cropping the car according to the specified bbox
    -- Adding 1 to account for the fact that Torch indexing is 1-based
    -- Also, note that we're not doing cx+cw-1 (since we're using a 1-based index)
    carImg = image.crop(cimg, cx+1, cy+1, cx+cw, cy+ch)
    -- Scaling the image to the input resolution
    scImg = image.scale(carImg, 64, 64)
    
    -- Creating the input tensor
    input = torch.Tensor(1, 3, 64, 64);
    input[1] = scImg;
    
    -- Getting output from the network
    local output = model:forward(input:cuda())
    -- Output is a table of 16 heatmaps, two from each hourglass. The 15th entry is what we need.
    -- The other entries correspond to predictions that are either from a lower layer, or predictions 
    -- that are not necessary.
    if type(output) == 'table' then
        output = output[#output]
    end

    -- Saving the predictions for the image
    -- saved.input[i]:copy(input[1])
    
    -- Obtain the keypoints from the output heatmaps from the network
    keyPoints = postprocess(output, outputRes);

    -- Copy them to the 'saved' tensor, to write them to an output file
    saved.preds[i]:copy(keyPoints[1])
    
    -- Increment the index
    i = i + 1;

end


-- Write the predictions to the output text file
fd = io.open(resultPath, 'w')
for i = 1, numImgs do
    -- Write the keypoint X and Y coordinates (1-based) and the confidence scores (comma-separated)
	for j = 1,numKps do
		fd:write(tostring(saved.preds[i][j][1])..','..tostring(saved.preds[i][j][2])..','..tostring(saved.preds[i][j][5]))
		if j ~= numKps then
			fd:write(tostring(','))
		end
	end
	fd:write(tostring('\n'))
end
