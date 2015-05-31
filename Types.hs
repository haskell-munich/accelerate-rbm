
module Types where

import Data.Array.Accelerate(Array, DIM1, DIM2, DIM3)

-- the weight matrix
type W = Array DIM2 Float

type Bias = Array DIM1 Float
-- the biases for the visible units
type V = Bias

-- the biases for the hidden units
type H = Bias

data RBM = RBM { nv :: Int -- number of visibles units
               , nh :: Int -- number of hidden units
               , minibatchSize :: Int
               , weights :: W
               , vbias :: V
               , hbias :: H }
           deriving (Show)


type Act = Array DIM2 Float
type Probs = Array DIM2 Float
type State = Array DIM2 Bool

-- the activations of the visible units
type VAct = Act

-- the probabilities of the hidden units
type VProbs = Probs

-- the states of the visible units
type VState = State


-- the activations of the hidden units
type HAct = Act

-- the probabilities of the hidden units
type HProbs = Probs

-- the states of the hidden units
type HState = State
