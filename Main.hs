module Main where

import Prelude hiding (replicate, zipWith, map, unzip)
import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, Exp, DIM1, DIM2, use, lift, replicate, All(..), zipWith,
   transpose, fold, fromList, map, unzip,
   Int64, Int32, lift)

import Data.Bits(Bits((.&.)))
   
import Data.Array.Accelerate.Interpreter as I

import Types


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


type IntR = Int64
type PRNG = Array DIM1 IntR
type Randoms = Array DIM1 IntR

-- Generate numbers from 0 to 1023 and also a new state
-- that can be used to generate more random numbers.
randoms :: Acc PRNG -> (Acc PRNG, Acc Randoms)
randoms state = (map next state,
                 map gen state)
  where next :: Exp IntR -> Exp IntR
        next x = x + (x `div` 2) + (x `div` 128)
        gen :: Exp IntR -> Exp IntR
        gen x = (x .&. 1023) --  1023.0


testRandoms :: IO ()
testRandoms =
  do let n = 3
         r0 = fromList (Z :. n) [123678, 123789, 234890]
              :: Array DIM1 IntR
         (r1, as) = randoms (use r0)
         (r2, bs) = randoms r1
         (r3, cs) = randoms r2
         (r4, ds) = randoms r3
         (r5, es) = randoms r4
     print (I.run as)
     print (I.run bs)
     print (I.run cs)
     print (I.run ds)
     print (I.run es)
  
  
testRBM =
  let (nv, nh) = (3, 4)
      rbm = initialWeights nv nh
      v1 = fromList (Z :. nv) [0,1,1] :: VState
      h1act = I.run $ hact rbm v1
  in
   print h1act



main =
  -- testRBM
  testRandoms
