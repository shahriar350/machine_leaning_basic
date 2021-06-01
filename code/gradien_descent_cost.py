import numpy as np

iteration = 100
learning_rate = 0.02


def gradient_decent(x, y):
    n = len(x)
    slop_current = intercept_current = 0
    for i in range(iteration):
        # now find the cost (1/n) sum ( y_original - y_predicted)**2
        # y_predicted = mx + c
        # first calculate y_predicted
        y_predicted = (slop_current * x) + intercept_current
        # now calculate cost function

        # now calculate derivation of slop = - (2/n) * sum( x (y_original-y_predicted))
        slop_derivation = - (2 / n) * sum(x * (y - y_predicted))
        # now calculate intercept_derivation = - (2/n) * sum(y-y_predicted)
        intercept_derivation = - (2 / n) * sum(y - y_predicted)

        # we calculate current slop and intercept
        # now calculate next slop and intercept
        slop_current = slop_current - learning_rate * slop_derivation
        intercept_current = intercept_current - (learning_rate * intercept_derivation)

        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        print(f'Iteration: {i} : cost: {cost} - slop_current: {slop_current} - intercept_current: {intercept_current}')

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_decent(x, y)