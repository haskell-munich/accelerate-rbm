import Prelude hiding (replicate, zipWith)
import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, DIM1, DIM2, use, lift, replicate, All(..), zipWith,
   transpose, fold)
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

-- sigmoid

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


-- propup -- p(h|v)
  
-- sampleH -- sample from p(h|v)
-- propdown -- p(v|h)
-- sampleV -- sample from p(v|h)
-- cd1

main =
  print $ initialWeights 3 4
