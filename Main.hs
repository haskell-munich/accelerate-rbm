module Main where

import Prelude as P
import Data.Array.Accelerate as A

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



sigmoid :: Exp Float -> Exp Float
sigmoid act =
  1 / (1 + exp (-act))


-- calculate the activations of the hiddens units given states of the
-- visible units. All rows of the matrix that correspond to an
-- activated visible are summed up.
hact :: RBM -> Acc VState -> Acc HAct
hact (RBM nv nh w v h) vis =
  A.zipWith (+)
     (use h)
     (fold (+) 0
       (transpose
         (A.zipWith rif
           repv
           (use w))))
  where
    repv :: Acc (Array DIM2 Bool)
    repv = (A.replicate (lift $ Z :. All :. nh) vis)
    rif :: Exp Bool -> Exp Float -> Exp Float
    rif v w = v ? (w, 0.0)

-- sample the state of the hiddens.
hsample :: Acc PRNG -> Acc HProbs -> (Acc PRNG, Acc HState)
hsample prng1 hprobs =
  let (prng2, rs) = randoms prng1
  in (prng2,
      A.zipWith (A.<*) rs hprobs)

-- propup -- p(h|v)
propup :: Acc PRNG -> RBM -> Acc VState -> (Acc PRNG, Acc HState)
propup prng rbm vis =
  let hprops = A.map sigmoid $ hact rbm vis
  in hsample prng hprops



-- calculate the activations of the visible units given states of the
-- hidden units. All columns of the matrix that correspond to an
-- activated hidden are summed up.
vact :: RBM -> Acc HState -> Acc VAct
vact (RBM nv nh w v h) hid =
  A.zipWith (+)
     (use v)
     (fold (+) 0
       (A.zipWith rif
          reph
          (transpose (use w))))
  where
    reph :: Acc (Array DIM2 Bool)
    reph = (A.replicate (lift $ Z :. nv :. All) hid)
    rif :: Exp Bool -> Exp Float -> Exp Float
    rif v w = v ? (w, 0.0)

-- sample the state of the visibles.
vsample :: Acc PRNG -> Acc VProbs -> (Acc PRNG, Acc VState)
vsample prng1 vprobs =
  let (prng2, rs) = randoms prng1
  in (prng2,
      A.zipWith (A.<*) rs vprobs)

-- propdown -- p(v|h)
propdown :: Acc PRNG -> RBM -> Acc HState -> (Acc PRNG, Acc VState)
propdown prng rbm hid =
  let vprops = A.map sigmoid $ vact rbm hid
  in vsample prng vprops




  

type IntR = Int64
type PRNG = Array DIM1 IntR
type Randoms = Array DIM1 Float


mkPRNG :: Int -> IO PRNG
mkPRNG size =
  do let mx = maxBound :: IntR
     rs <- replicateM size (getStdRandom (randomR (20000, mx)))
     return $ fromList (Z :. size) rs
  
-- Generate numbers from 0.0 to 1.0 and also a new state
-- that can be used to generate more random numbers.
randoms :: Acc PRNG -> (Acc PRNG, Acc Randoms)
randoms state = (A.map next state,
                 A.map gen state)
  where next :: Exp IntR -> Exp IntR
        next x = x + (x `div` 2) + (x `div` 128)
        gen :: Exp IntR -> Exp Float
        gen x = A.fromIntegral (x .&. 4095) / 4096.0


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
     rv1 <- mkPRNG nv
     rh1 <- mkPRNG nh
     let rbm = initialWeights nv nh
         v1 = fromList (Z :. nv) [False, True, True] :: VState
         h1act = I.run $ hact rbm (use v1)
         (rh2', h1s') = propup (use rh1) rbm (use v1)
         (rh2, h1s) = (I.run rh2', I.run h1s')
         (rv2', v2s') = propdown (use rv1) rbm (use h1s)
         (rv2, v2s) = (I.run rh2', I.run v2s')
     print h1act
     print h1s
     print v2s



main =
  do testRandoms
     testRBM
