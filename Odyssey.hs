module Odyssey(load) where

import Data.List as L
import Data.Map as M
import Data.Char as CH

chars = "_" ++ ['a' .. 'z'] ++ " .,;:-"

charsMap = M.fromList (zip chars [0..])

load :: Int -> IO ([Char], [[Char]], [[Int]])
load len =
  do content <- readFile "pg1727-part.txt"
     let chdata =
           L.filter minlen $
           L.map (take len) $
           L.tails content
     return (chars,
             chdata,
             L.map (L.map encode) chdata)
  where
    minlen x = length x == len
    encode ch =
      case M.lookup (CH.toLower ch) charsMap of
        Nothing -> 0
        Just k -> k

