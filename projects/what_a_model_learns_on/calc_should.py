def calc_should(feature, y, regular=0):
    """
    The function gets a certain binary feature column from a dataset and the label vector,
    and returns the "should" score of that feature, which is the gain of an XGBoost
    tree node that splits according to the feature.

    regular is an optional parmeter that allows for adding regularization to the calculation
    of the similarity score.
    """

    def calc_similarity(x, regular=0):
        return ((x - 0.5).sum()) ** 2 / (len(x) + regular)

    assert (np.sort(feature.unique()) == [0, 1]).all(), "feature should only have 0s and 1s"
    assert len(feature) == len(y), "The dataframe and the label vector are not the same length"


    zeros = y[feature == 0]
    ones = y[feature == 1]

    zeros_similarity = calc_similarity(zeros, regular)
    ones_similarity = calc_similarity(ones, regular)
    root_similarity = calc_similarity(y, regular)

    gain = zeros_similarity + ones_similarity - root_similarity

    return gain