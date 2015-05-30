module Main where

import Prelude as P
import Data.Array.Accelerate as A

import System.Random(getStdRandom, randomR)
import Control.Monad(replicateM)

import Data.Bits(Bits((.&.)), xor)
   
import Data.Array.Accelerate.Interpreter as I

import Types

import Odyssey


initialWeights :: Int -> Int -> RBM
initialWeights nv nh =
  let w = I.run $ fill (constant $ Z :. nv :. nh) 0.0 :: W
      v = I.run $ fill (constant $ Z :. nv) 0.0 :: V
      h = I.run $ fill (constant $ Z :. nh) 0.0 :: H
  in RBM nv nh w v h



sigmoid :: Exp Float -> Exp Float
sigmoid act =
  1 / (1 + exp (-act))


-- sample the state of the hiddens.
hsample :: Acc PRNG -> Acc HProbs -> (Acc PRNG, Acc HState)
hsample prng1 hprobs =
  let (prng2, rs) = randoms prng1
  in (prng2,
      A.zipWith (A.<*) rs hprobs)

-- propup -- p(h|v), sampled
propup :: Acc PRNG -> RBM -> Acc VState -> (Acc PRNG, Acc HState)
propup prng rbm vis =
  hsample prng $ propupP rbm vis

-- propup -- p(h|v), the probabilities
propupP :: RBM -> Acc VState -> Acc HProbs
propupP rbm vis =
  A.map sigmoid $ hact rbm vis



{- johannes drever WIP

-- calculate the activations of the visible units given states of the
-- hidden units. All columns of the matrix that correspond to an
-- activated hidden are summed up.
vact :: RBM -> Acc HState -> Acc VAct
vact (RBM nv _ w v _) = act nv (transpose $ use w) (use v)

-- calculate the activations of the hiddens units given states of the
-- visible units. All rows of the matrix that correspond to an
-- activated visible are summed up.
hact :: RBM -> Acc VState -> Acc HAct
hact (RBM _ nh w _ h) = act nh (use w) (use h) 

act :: Int -> Acc W -> Acc Bias -> Acc State -> Acc Act
act nout w bias inputs =
     A.zipWith (+)
    bias
     (fold (+) 0
       (A.zipWith rif
          reph
          w))
  where
    reph :: Acc (Array DIM2 Bool)
    reph = (A.replicate (lift $ Z :. nout :. All) inputs)
    rif :: Exp Bool -> Exp Float -> Exp Float
    rif v w = v ? (w, 0.0)
-}


-- calculate the activations of the visible units given states of the
-- hidden units. All columns of the matrix that correspond to an
-- activated hidden are summed up.
vact :: RBM -> Acc HState -> Acc VAct
vact (RBM nv nh w v h) hid =
  A.zipWith (+)
     (use v)
     (fold (+) 0
        (transpose
          (A.zipWith rif
            reph
            (transpose (use w)))))
  where
    reph :: Acc (Array DIM2 Bool)
    reph = (A.replicate (lift $ Z :. All :. nv) hid)
    rif :: Exp Bool -> Exp Float -> Exp Float
    rif v w = v ? (w, 0.0)

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


-- sample the state of the visibles.
vsample :: Acc PRNG -> Acc VProbs -> (Acc PRNG, Acc VState)
vsample prng1 vprobs =
  let (prng2, rs) = randoms prng1
  in (prng2,
      A.zipWith (A.<*) rs vprobs)

-- propdown -- p(v|h)
propdown :: Acc PRNG -> RBM -> Acc HState -> (Acc PRNG, Acc VState)
propdown prng rbm hid =
  let vprobs = A.map sigmoid $ vact rbm hid
  in vsample prng vprobs


data CD1PRNGS =
  CD1PRNGS { vprng :: PRNG,
             hprng :: PRNG }

mkCD1PRNGS rbm =
  do let (numv, numh) = (nv rbm, nh rbm)
     rv <- mkPRNG numv
     rh <- mkPRNG numh
     return $ CD1PRNGS rv rh

-- update the weights according to CD-1
cd1 :: Float -> CD1PRNGS -> RBM -> VState
       -> (CD1PRNGS, RBM, Scalar Float)
cd1 learningRate rn rbm vis1 =
  let (rnh1, hid1) = propup (use $ hprng rn) rbm (use vis1)
      (rnv1, vis2) = propdown (use $ vprng rn) rbm hid1
      hid2p = propupP rbm vis2
      (numv, numh) = (nv rbm, nh rbm)

      -- find visibles and hiddens that are both active
      d_data :: Acc W
      d_data =
        A.zipWith mulBB
          (A.replicate (lift $ Z :. All :. numh) (use vis1))
          (A.replicate (lift $ Z :. numv :. All) hid1)
      mulBB :: Exp Bool -> Exp Bool -> Exp Float
      mulBB a b = a&&*b ? (1.0, 0.0)

      -- find visibles and hiddens that are both active


      dr1 = (A.replicate (lift $ Z :. All :. numh) vis2)
      dr2 = (A.replicate (lift $ Z :. numv :. All) hid2p)
        
      d_recon :: Acc W
      d_recon =
        A.zipWith mulBP
          (A.replicate (lift $ Z :. All :. numh) vis2)
          (A.replicate (lift $ Z :. numv :. All) hid2p)
      mulBP :: Exp Bool -> Exp Float -> Exp Float
      mulBP a b = a ? (b, 0.0)

      delta = A.zipWith (-) d_data d_recon

      upd :: Exp Float -> Exp Float -> Exp Float
      upd w d = w + constant learningRate * d

      updatedWeights = A.zipWith upd (use $ weights rbm) delta

      updatedVbias =
        A.zipWith (+) (use $ vbias rbm)
          (A.map mulLearningRate
             (A.zipWith diffbb (use vis1) vis2))

      diffbb :: Exp Bool -> Exp Bool -> Exp Float
      diffbb a b = A.fromIntegral (boolToInt a - boolToInt b)

      updatedHbias =
        A.zipWith (+) (use $ hbias rbm)
          (A.map mulLearningRate
             (A.zipWith diffbf hid1 hid2p))

      diffbf :: Exp Bool -> Exp Float -> Exp Float
      diffbf a b = A.fromIntegral (boolToInt a) - b

      mulLearningRate :: Exp Float -> Exp Float
      mulLearningRate d = d * constant learningRate


      -- the reconstruction error
      recerr = A.fold (+) 0 (A.zipWith bindiff (use vis1) vis2)

      bindiff :: Exp Bool -> Exp Bool -> Exp Float
      bindiff a b = a /=* b ? (1, 0)
      
  in (CD1PRNGS (I.run rnv1) (I.run rnh1),
      rbm { weights = I.run updatedWeights
          , vbias = I.run updatedVbias
          , hbias = I.run updatedHbias },
      I.run recerr)
     

  

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

testCD1 =
  do let (nv, nh) = (5, 3)
     let rbm1 = initialWeights nv nh
     rn1 <- mkCD1PRNGS rbm1
     let v1 = fromList (Z :. nv)
              [False, True, False, True, True] :: VState
     let (rn2, rbm2, recerr) = cd1 0.01 rn1 rbm1 v1
     print (rbm1)
     print (rbm2)

testOdysseyLetters =
  do let ngram = 3
     (chars, chdat, idat) <- Odyssey.load ngram
     testOdysseyLettersRun ngram chars idat

testOdysseyLettersRun :: Int -> [Char] -> [[Int]] -> IO ()
testOdysseyLettersRun ngram chars idat =
  do
     -- mapM_ print (P.zip chdat idat)
     print (nv, nh)
     let rbm1 = initialWeights nv nh
     rn1 <- mkCD1PRNGS rbm1
     learn 0 rbm1 rn1 idat
  where
    nchars = P.length chars
    (nv, nh) = (ngram*nchars, 50)
    encodeData :: Int -> Int -> [Int] -> Array DIM1 Bool
    encodeData nchars nv dat =
      let onehot d = [ if i == d then True else False
                     | i <- [0..nchars-1]]
      in fromList (Z :. nv) (P.concat $ P.map onehot dat) :: VState
    learn :: Int -> RBM -> CD1PRNGS -> [[Int]] -> IO ()
    learn rep rbm1 rn1 (dat:idat) =
      do let v1 = encodeData nchars nv dat
             (rn2, rbm2, recerr) = cd1 0.01 rn1 rbm1 v1
         print recerr
         if rep `mod` 10 == 9
           then reportHiddens ngram chars rbm1
           else return ()
         learn (succ rep) rbm2 rn2 idat


reportHiddens ngram chars rbm =
  sequence_ [ reportHidden h | h <- [0 .. nh rbm - 1]]
  where
    reportHidden h =
      do print("hidden", h)
         mapM_ print
           [ (g, ch, w)
           | g <- [0..ngram-1]
           , (ch, chi) <- P.zip chars [0..]
           , let w = indexArray (weights rbm)
                     (Z :. (nchars*g + chi) :. h)
           , w > 0.0 ]
    nchars = P.length chars


main =
  do testRandoms
     -- testRBM
     -- testCD1
     -- testOdysseyLetters
