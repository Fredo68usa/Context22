from sklearn.metrics import mean_absolute_percentage_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print (mean_absolute_percentage_error(y_true, y_pred))
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print (mean_absolute_percentage_error(y_true, y_pred))
print (mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7]))
# the value when some element of the y_true is zero is arbitrarily high because
# of the division by epsilon
y_true = [1., 0., 2.4, 7.]
y_pred = [1.2, 0.1, 2.4, 8.]
print (mean_absolute_percentage_error(y_true, y_pred))
