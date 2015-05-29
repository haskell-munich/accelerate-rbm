
module Main where

import Prelude hiding (replicate, zipWith)
import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, DIM1, DIM2, use, lift, replicate, All(..), zipWith,
   transpose, fold)
import Data.Array.Accelerate.Interpreter as I

import Types


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
