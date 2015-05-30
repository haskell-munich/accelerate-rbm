
module Types where

import Data.Array.Accelerate(Array, DIM1, DIM2)

-- the weight matrix
type W = Array DIM2 Float

-- the biases for the visible units
type V = Array DIM1 Float

-- the biases for the hidden units
type H = Array DIM1 Float

data RBM = RBM { nv :: Int -- number of visibles units
               , nh :: Int -- number of hidden units
               , weights :: W
               , vbias :: V
               , hbias :: H }
           deriving (Show)

-- the activations of the visible units
type VAct = Array DIM1 Float

-- the probabilities of the hidden units
type VProbs = Array DIM1 Float

-- the states of the visible units
type VState = Array DIM1 Bool


-- the activations of the hidden units
type HAct = Array DIM1 Float

-- the probabilities of the hidden units
type HProbs = Array DIM1 Float

-- the states of the hidden units
type HState = Array DIM1 Bool
