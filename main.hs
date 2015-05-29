import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, DIM1, DIM2, use)
import Data.Array.Accelerate.Interpreter as I

-- the weight matrix
type W = Array DIM2 Float

-- the biases for the visible units
type V = Array DIM1 Float

-- the biases for the hidden units
type H = Array DIM1 Float

data RBM = RBM { weights :: W
               , vbias :: V
               , hbias :: H }
           deriving (Show)

-- the activations of the visible units
type VAct = Array DIM1 Float

-- the states of the visible units
type VState = Array DIM1 Bool


initialWeights :: Int -> Int -> RBM
initialWeights nv nh =
  let w = I.run $ fill (constant $ Z :. nv :. nh) 0.0 :: W
      v = I.run $ fill (constant $ Z :. nv) 0.0 :: V
      h = I.run $ fill (constant $ Z :. nh) 0.0 :: H
  in RBM w v h

-- sigmoid

hact :: RBM -> VState -> Acc VAct
hact (RBM w v h) vis =
  use v

-- propup -- p(h|v)
-- hact :: RBM -> 
  
-- sampleH -- sample from p(h|v)
-- propdown -- p(v|h)
-- sampleV -- sample from p(v|h)
-- cd1

main =
  print $ initialWeights 3 4
