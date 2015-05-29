import Prelude hiding (replicate, zipWith, map, unzip)
import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, Exp, DIM1, DIM2, use, lift, replicate, All(..), zipWith,
   transpose, fold, fromList, map, unzip,
   Int64, lift)

import Data.Bits(Bits((.&.)))
   
import Data.Array.Accelerate.Interpreter as I

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

-- the states of the visible units -- should be bool
type VState = Array DIM1 Float

-- the activations of the hidden units
type HAct = Array DIM1 Float

-- the states of the hidden units -- should be bool
type HState = Array DIM1 Float


initialWeights :: Int -> Int -> RBM
initialWeights nv nh =
  let w = I.run $ fill (constant $ Z :. nv :. nh) 0.0 :: W
      v = I.run $ fill (constant $ Z :. nv) 0.0 :: V
      h = I.run $ fill (constant $ Z :. nh) 0.0 :: H
  in RBM nv nh w v h


-- calculate the activations of the hiddens units given states of the
-- visible units. All columns of the matrix that correspond to an
-- activated visible are summed up.
hact :: RBM -> VState -> Acc HAct
hact (RBM nv nh w v h) vis =
  zipWith (+)
     (use h)
     (fold (+)
       0
       (transpose
         (zipWith (*)
           (repv :: Acc W)
           ((use w) :: Acc W))))
  where
    repv :: Acc (Array DIM2 Float)
    repv = (replicate (lift $ Z :. All :. nh) (use vis))


sigmoid :: Exp Float -> Exp Float
sigmoid act =
  1 / (1 + exp (-act))

-- propup -- p(h|v)
propup :: RBM -> VState -> Acc HState
propup rbm vis =
  map sigmoid $ hact rbm vis

  
-- sampleH -- sample from p(h|v)
-- propdown -- p(v|h)
-- sampleV -- sample from p(v|h)
-- cd1


type PRNG = Array DIM1 Int64
type Randoms = Array DIM1 Float

-- Generate numbers from 0.0 to 1.0 and also a new state
-- that can be used to generate more random numbers.
randoms :: Acc PRNG -> (Acc PRNG, Acc Randoms)
randoms state = (map next state,
                 map gen state)
  where next :: Exp Int64 -> Exp Int64
        next x = x + (x `div` 2) + (x `div` 128)
        gen :: Exp Int64 -> Exp Float
        gen x = fromIntegral (x .&. 1023) / 1023.0

main =
  let (nv, nh) = (3, 4)
      rbm = initialWeights nv nh
      v1 = fromList (Z :. nv) [0,1,1] :: VState
      h1act = I.run $ hact rbm v1
--      h1s = 
  in
   print h1act

