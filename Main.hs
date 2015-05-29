module Main where

import Prelude hiding (replicate, zipWith, map, unzip, (<*))
import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..),
   Array, Exp, DIM1, DIM2, use, lift, replicate, All(..), zipWith,
   transpose, fold, fromList, map, unzip,
   Int64, Int32, lift, (<*))

import System.Random(getStdRandom, randomR)
import Control.Monad(replicateM)

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

hsample :: Acc PRNG -> Acc HProbs -> (Acc PRNG, Acc HState)
hsample prng1 hprobs =
  let (prng2, rs) = randoms prng1
  in (prng2,
      zipWith (<*)
        (map (floor . (*1024)) hprobs)
        rs)

-- propup -- p(h|v)
propup :: Acc PRNG -> RBM -> VState -> (Acc PRNG, Acc HState)
propup prng rbm vis =
  do let hprops = map sigmoid $ hact rbm vis
     hsample prng hprops

  
-- sampleH -- sample from p(h|v)
-- propdown -- p(v|h)
-- sampleV -- sample from p(v|h)
-- cd1


type IntR = Int64
type PRNG = Array DIM1 IntR
type Randoms = Array DIM1 IntR


mkPRNG :: Int -> IO PRNG
mkPRNG size =
  do let mx = maxBound :: IntR
     rs <- replicateM size (getStdRandom (randomR (20000, mx)))
     return $ fromList (Z :. size) rs
  
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
     let f _r 0 = return ()
         f r0 num =
           do let (r1, as) = randoms $ use r0
              print (I.run as)
              f (I.run r1) (pred num)
     r0 <- mkPRNG n
     print("r0", r0)
     f r0 10
  
  
testRBM =
  do let (nv, nh) = (3, 4)
     rv <- mkPRNG nv
     rh1 <- mkPRNG nh
     let rbm = initialWeights nv nh
         v1 = fromList (Z :. nv) [0,1,1] :: VState
         h1act = I.run $ hact rbm v1
         (rh2', h1s') = propup (use rh1) rbm v1
         (rh2, h1s) = (I.run rh2', h1s')
     print h1act
     print h1s



main =
  -- testRandoms
  testRBM
