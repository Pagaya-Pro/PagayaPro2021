from typing import List
import numpy as np


def compute_weighted_average(x: List[float],
                             w: List[float])-> float:
    return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w) if sum(w) !=0 else np.nan

def check_compute_weighted_average(x:List[float],
                                   w:List[float])-> float:
    answer = compute_weighted_average(x, w)

    print(f"The weighted average of {x} with weights {w} is {answer}")
    return answer

def main():
    check_compute_weighted_average([1,2,3],[4,5,6])
    check_compute_weighted_average([1, 0 ,1], [1, -1, 0])
    check_compute_weighted_average([5,5,5], [0, 2, 4])

if __name__=="__main__":
    main()