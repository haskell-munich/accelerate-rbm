import Data.Array.Accelerate
  (fill, constant, Acc, Z(..), (:.)(..), Array, DIM1, DIM2)
import Data.Array.Accelerate.Interpreter as I

type W = Array DIM2 Float
type V = Array DIM1 Float
type H = Array DIM1 Float

data RBM = RBM { weights :: Acc W
               , vbias :: Acc V
               , hbias :: Acc H }

initialWeights :: Int -> Int -> RBM
initialWeights nv nh =
  let w = fill (constant $ Z :. nv :. nh) 0.0 :: Acc W
      v = fill (constant $ Z :. nv) 0.0 :: Acc V
      h = fill (constant $ Z :. nh) 0.0 :: Acc H
  in RBM w v h

runRBM (RBM w v h) =
  (I.run w, I.run v, I.run h)

-- sigmoid
-- propup -- p(h|v)
-- sampleH -- sample from p(h|v)
-- propdown -- p(v|h)
-- sampleV -- sample from p(v|h)
-- cd1

main =
  print $ runRBM $ initialWeights 3 4
