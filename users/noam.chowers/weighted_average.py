from typing import List


def compute_weighted_average(x: List[float],
                             w: List[float])-> float:
    try:
        return sum([x1 * w1 for x1, w1 in zip(x, w)]) / sum(w)
    except ZeroDivisionError:
        return 'Error'

def check_compute_weighted_average(x:List[float],
                                   w:List[float])-> float:
    ans = compute_weighted_average(x, w)
    print(f"The weighted average of {x} with weights {w} is {ans}")
    return ans

def main():
    try:
        check_compute_weighted_average([1,2,3],[4,5,6])
    except:
        print('Error')
    try:
        check_compute_weighted_average([1, 0 ,1], [1, -1, 0])
    except:
        print('Error')
    try:
        check_compute_weighted_average([5,5,5], [0, 2, 4])
    except:
        print('Error')

if __name__=="__main__":
    main()