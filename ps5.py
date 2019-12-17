# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Deniz Sert
# Collaborators: Frank Gonzalez, Karen Gao
# Time: 5 hrs
#Late Days: 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re
# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2000)
TESTING_INTERVAL = range(2000, 2017)

"""
Begin helper code
"""
class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

##########################
#    End helper code     #
##########################

def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    
#    mu for x
    m_x = 0
    tot_x = 0
    for i in x:
        tot_x += i
    m_x = tot_x/len(x)
    
    #mu for y
    m_y = 0
    tot_y = 0
    for i in y:
        tot_y += i
    m_y = tot_y/len(y)
    
    #slope equation
    slope = 0
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i]-m_x)*(y[i]-m_y)
    for j in range(len(x)):
        denominator += (x[j]-m_x)**2
    
    
    slope = numerator/denominator
    
    b = m_y - (slope*m_x)
    
    
    return slope, b
    
    


def get_squared_error(x, y, m, b):
    '''
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    '''
    #predicted y^hat vals using regression eq
    predicted = [m*i+b for i in x]
    
    #residuals
    errors = [y[i]-predicted[i] for i in range(len(y))]
    #sqare residuals
    sq_errors = [errors[i]**2 for i in range(len(errors))]
    #sum squared residuals
    sq_errors_sum = sum(sq_errors)
    
    return sq_errors_sum
    
    
    
#    def rSquared(observed, predicted):
#    error = ((predicted - observed)**2).sum()
#    meanError = error/len(observed)
#    return 1 - (meanError/np.var(observed))


def generate_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    #note that the degrees argument is a list
    array_list = []
    for i in range(len(degrees)):
        array_list.append(np.polyfit(x, y, degrees[i]))
    return array_list
    


def evaluate_models_on_training(x, y, models, display_graphs):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        Degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and standard error/slope (if this model is linear).

    R-squared and standard error/slope should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    r_sqs = []
    for model in models:
        #find r_sq and append to list of r_sqs
        predicted = np.polyval(model, x)
        r_sq = r2_score(y, predicted)
        r_sq = round(r_sq, 4)
        r_sqs.append(r_sq)
        
        #generates graph
        if display_graphs:
            plt.plot(x, y, 'b.')
            if len(model) == 2:
                title = 'R^2 = ' + str(r_sq) + "  Std error: " + str(standard_error_over_slope(x, y, predicted, model)) + "  Degree: " + str(len(model)-1)
            else:
                title = 'R^2 = ' + str(r_sq) + "  Degree: " + str(len(model)-1)
            plt.title(title)
            plt.xlabel("Time (years)")
            plt.ylabel("Temperature (C)")
            plt.plot(x, predicted, color = 'red')
            plt.show()
    return r_sqs


def generate_cities_averages(dataset, cities, years):
    """
    For each year in the given range of years, computes the average of the
    annual temperatures in the given cities.

    Args:
        dataset: instance of Dataset
        cities: a list of the names of cities to include in the average
            annual temperature calculation
        years: a list of years to evaluate the average annual temperatures at

    Returns:
        a 1-d numpy array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    avg_temps_final = ([])
    for year in years:
        all_city_temps = 0
        for city in cities:
            temps = dataset.get_daily_temps(city, year)
            temps_sum = sum(temps)
            avg_temp = temps_sum/len(temps)
            all_city_temps += avg_temp
        avg_temp = all_city_temps/len(cities)
        avg_temps_final.append(avg_temp)

    return avg_temps_final


def find_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have the same slope (within the
        acceptable tolerance), (2,5) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
 
            
    best_slope = 0
    best_start_index = 0
    for i in range(0, len(x) - length + 1):
        slope = linear_regression(x[i:(i+length)], y[i:(i+length)])[0]
        if positive_slope:
            if slope > best_slope and abs(slope - best_slope) > 10e-8:
                best_slope = slope
                best_start_index = i
        else:
            if slope < best_slope and abs(slope - best_slope) > 10e-8:
                best_slope = slope
                best_start_index = i
    if best_slope == 0:
        return None
    else:
#        print("Slope: " , best_slope)
        return (best_start_index, best_start_index + length)
                


def get_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    diffs = []
    #y - y^hat
    for i in range(len(y)):
        diffs.append(y[i]-estimated[i])
    #(y-y^hat)**2
    for i in range(len(diffs)):
        diffs[i] = diffs[i]**2
    #sum of sqaures
    sum_diffs = sum(diffs)
    
    #final formula
    return (sum_diffs/len(diffs))**(1/2)


def evaluate_models_on_testing(x, y, models, display_graphs):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    rsmes = []
    for model in models:
        #find r_sq and append to list of r_sqs
        predicted = np.polyval(model, x)
        rsme = round(get_rmse(y, predicted), 4)
        rsmes.append(rsme)
        
#        for i in range(1, 10):
#            if len(model) == i:
#                degree = i-1
        
        #generates graph
        if display_graphs:
            plt.plot(x, y, 'b.')
            title = 'RSME ' + str(rsme) + "  Degree: " + str(len(model)-1)
            plt.title(title)
            plt.xlabel("Time (years)")
            plt.ylabel("Temperature (C)")
            plt.plot(x, predicted, color = 'red')
            plt.show()
    return rsmes



if __name__ == '__main__':

    pass

#    # Problem 4A
#    data = Dataset('data.csv')
#    y = np.array([data.get_temp_on_date('PORTLAND', 12, 25, year) for year in range(1961, 2016)])
#    x = np.array([i for i in range(1961, 2016)])
#    
#    model = generate_models(x, y, [1])
#    evaluate_models_on_training(x, y, model, display_graphs=True)
#    
#    # Problem 4B
#    years = [i for i in range(1961, 2017)]
#    data = Dataset('data.csv')
#    x = np.array([i for i in range(1961, 2017)])
#    y = np.array([])
#    
#
#    y = np.array(generate_cities_averages(data, ['PORTLAND'], years))
#   
#    
#    model = generate_models(x, y, [1])
#    evaluate_models_on_training(x, y, model, display_graphs=True)
#    
    
    
    # Problem 5B
#    data = Dataset('data.csv')
#    x = np.array([i for i in range(1961, 2017)])
#    y = np.array(generate_cities_averages(data, ['PHOENIX'], years))
#    
#    trend = find_trend(x, y, 30, True)
#    years = [i for i in range(1961, 2017)]
#    
#    start, end = trend[0], trend[1]
#    
#    print("start: " + str(start))
#    print("end: " + str(end))
#    
#    model = generate_models(x[start:end], y[start:end], [1])
#    evaluate_models_on_training(x, y, model, display_graphs=True)
#    
    # Problem 5C
#    data = Dataset('data.csv')
#    x = np.array([i for i in range(1961, 2017)])
#    y = np.array(generate_cities_averages(data, ['PHOENIX'], years))
#    
#    trend = find_trend(x, y, 15, False)
#    years = [i for i in range(1961, 2017)]
#    
#    start, end = trend[0], trend[1]
#    
##    print("start: " + str(start))
##    print("end: " + str(end))
##    
#    model = generate_models(x[start:end], y[start:end], [1])
#    evaluate_models_on_training(x, y, model, display_graphs=True)

    # Problem 6B
    #training data
#    data = Dataset('data.csv')
#    x = np.array([i for i in TRAINING_INTERVAL])
#    y = np.array(generate_cities_averages(data, CITIES, x))
#    
#    model = generate_models(x, y, [2, 10])
#    evaluate_models_on_training(x, y, model, display_graphs=True)
#    
#    
#    #testing
#    x = np.array([i for i in TESTING_INTERVAL])
#    y = np.array(generate_cities_averages(data, CITIES, x))
#    model = generate_models(x, y, [2, 10])
#    evaluate_models_on_testing(x, y, model, display_graphs = True)
    
    