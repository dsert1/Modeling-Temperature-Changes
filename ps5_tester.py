import unittest
import numpy as np
import math
import warnings

import ps5

class TestPS5(unittest.TestCase):

    def test_linear_regression(self):
        #simple y = x case
        x = [1,2,3,4,5,6,7,8,9,10]
        y = [1,2,3,4,5,6,7,8,9,10]
        (m, b) = ps5.linear_regression(x, y)
        self.assertEqual(m, 1, "Calculated slope is incorrect")
        self.assertEqual(b, 0, "Calculated intercept is incorrect")

        #y = 5x - 3 case
        x = [-5, -3, -1, 1, 2, 6, 10]
        y = [-28, -18, -8, 2, 7, 27, 47]
        (m, b) = ps5.linear_regression(x, y)
        self.assertEqual(m, 5, "Calculated slope is incorrect")
        self.assertEqual(b, -3, "Calculated intercept is incorrect")

        #not exact line case - also tests for float handling
        x = [0,1,2,3,4,5,6,7,8,9,10,11]
        y = [-10,2,1,5,12,13,15,14,19,25,30,29]
        (m, b) = ps5.linear_regression(x, y)
        self.assertTrue(math.isclose(3.241, m, rel_tol=1e-3), "Calculated slope is incorrect")
        self.assertTrue(math.isclose(-4.910, b, rel_tol=1e-3), "Calculated intercept is incorrect")


    def test_get_squared_error(self):
        #easy case
        x = [1,2,3,4,5,6,7,8,9,10]
        y = [1,2,3,4,5,6,7,8,9,10]
        m = 1
        b = 0
        sqe = ps5.get_squared_error(x, y, m ,b)
        self.assertEqual(sqe, 0, "Squared Error incorrect")

        #interesting case
        x = [0,1,2,3,4,5,6,7,8,9,10,11]
        y = [-10,2,1,5,12,13,15,14,19,25,30,29]
        m = 3
        b = -5
        sqe = ps5.get_squared_error(x, y, m ,b)
        self.assertEqual(sqe, 119, "Squared Error incorrect")


    def test_generate_models(self):
        degs_msg = "make_models should return one model for each given degree"
        list_type_msg = "make_models should return a list of models"
        array_type_msg = "each model returned by make_models should be of type np.array"
        coefficient_mismatch = "coefficients of returned model are not as expected"

        # simple y = x case.
        x = np.array(range(50))
        y = np.array(range(50))
        degrees = [1]
        models = ps5.generate_models(x, y, degrees)

        self.assertEqual(len(models), len(degrees), degs_msg)
        self.assertIsInstance(models, list, list_type_msg)
        self.assertIsInstance(models[0], np.ndarray, array_type_msg)
        self.assertListEqual(list(models[0]), list(np.polyfit(x, y, 1)), coefficient_mismatch)

        # two models for y = 2x case
        y = np.array(range(0,100,2))
        degrees = [1, 2]
        models = ps5.generate_models(x, y, degrees)
        self.assertEqual(len(models), len(degrees), degs_msg)
        self.assertIsInstance(models, list, list_type_msg)
        for m in models:
            self.assertIsInstance(m, np.ndarray, array_type_msg)
        for i in range(2):
            self.assertListEqual(list(models[i]), list(np.polyfit(x,y, degrees[i])), coefficient_mismatch)

        # three models
        degrees = [1,2,20]
        models = ps5.generate_models(x, y, degrees)
        self.assertEqual(len(models), len(degrees), degs_msg)
        self.assertIsInstance(models, list, list_type_msg)
        for m in models:
            self.assertIsInstance(m, np.ndarray, array_type_msg)
        for i in range(3):
            self.assertListEqual(list(models[i]), list(np.polyfit(x,y, degrees[i])), coefficient_mismatch)


    def test_generate_cities_averages(self):
        # test for just one city
        climate = ps5.Dataset('data.csv')
        test_years = np.array(range(2008, 2017))
        result = ps5.generate_cities_averages(climate, ['SEATTLE'], test_years)
        correct = [10.798633879781422, 11.218767123287673, 11.514383561643836, 10.586849315068493,
                    11.283196721311475, 12.106438356164384, 12.82917808219178, 13.13178082191781, 12.500546448087432]
        self.assertTrue(len(correct) == len(result), "Expected length %s, was length %s" % (len(correct), len(result)))

        for index in range(len(correct)):
            good_enough = math.isclose(correct[index], result[index])
            self.assertTrue(good_enough, "City averages do not match expected results")

        # national avg check (all cities)
        result = ps5.generate_cities_averages(climate, ps5.CITIES, test_years)
        correct = [16.130025884383087, 16.116286950252345, 16.334981975486663, 16.46957462148522,
                    17.173878343399483, 16.25620043258832, 16.47222062004326, 17.17817591925018, 17.198259994247916]
        self.assertTrue(len(correct) == len(result), "Expected length %s, was length %s" % (len(correct), len(result)))

        for index in range(len(correct)):
            good_enough = math.isclose(correct[index], result[index])
            self.assertTrue(good_enough, "City averages do not match expected results")

        # two-city check
        result = ps5.generate_cities_averages(climate, ['TAMPA', 'DALLAS'], test_years)
        correct = [21.76502732240437, 21.245616438356162, 20.80404109589041, 22.039109589041097,
                    22.272062841530055, 21.3113698630137, 20.88123287671233, 22.077945205479455, 22.181557377049177]
        self.assertTrue(len(correct) == len(result), "Expected length %s, was length %s" % (len(correct), len(result)))

        for index in range(len(correct)):
            good_enough = math.isclose(correct[index], result[index])
            self.assertTrue(good_enough, "City averages do not match expected results")

    def test_get_rmse(self):
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        estimate = [1, 4, 9, 16, 25, 36, 49, 64, 81]
        result = ps5.get_rmse(np.array(y), np.array(estimate))
        correct = 35.8515457593
        self.assertTrue(math.isclose(correct, result), "RMSE value incorrect")

        y = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        estimate = [1, 4, 9, 16, 25, 36, 49, 64, 81]
        result = ps5.get_rmse(np.array(y), np.array(estimate))
        correct = 40.513372278
        self.assertTrue(math.isclose(correct, result), "RMSE value incorrect")

    def test_evaluate_models_on_training(self):
        x = np.array(range(50))
        y = np.array(range(0,100,2))
        degrees = [1, 2]
        models = ps5.generate_models(x, y, degrees)
        r2 = ps5.evaluate_models_on_training(x, y, models, False)
        correct_r2 = [1,1]
        self.assertEqual(len(r2), len(correct_r2), "Returned incorrect r^2 values")
        for index in range(len(correct_r2)):
            good_enough = math.isclose(correct_r2[index], r2[index])
            self.assertTrue(good_enough, "Returned incorrect r^2 values")


    def test_find_trend(self):
        # Test 1: Existing positive and negative slope intervals on city data
        temp = ps5.Dataset('data.csv')
        test_years = np.array(range(1961, 2016))
        yearly_temps = ps5.generate_cities_averages(temp, ['PORTLAND'], test_years)

        result_neg = ps5.find_trend(test_years, yearly_temps, 20, False)
        correct_start = 31
        correct_end = 51
        self.assertIsNotNone(result_neg, "Returned None, but valid interval exists.")
        self.assertEqual(correct_start, result_neg[0], "Start year incorrect")
        self.assertEqual(correct_end, result_neg[1], "End year incorrect")

        result_pos = ps5.find_trend(test_years, yearly_temps, 20, True)
        correct_start = 15
        correct_end = 35
        self.assertIsNotNone(result_pos, "Returned None, but valid interval exists.")
        self.assertEqual(correct_start, result_pos[0], "Start year incorrect")
        self.assertEqual(correct_end, result_pos[1], "End year incorrect")

        # Test 2: y = 2x
        x = np.array(range(50))
        y = np.array(range(0,100,2))
        result_pos = ps5.find_trend(x, y, len(x)//2, True)
        result_neg = ps5.find_trend(x, y, len(x)//2, False)
        pos_correct_start = 0
        pos_correct_end = len(x)//2

        self.assertIsNone(result_neg, "Returned an interval, but should be None.")

        self.assertIsNotNone(result_pos, "Returned None, but valid interval exists.")
        self.assertEqual(pos_correct_start, result_pos[0], "Start year incorrect")
        self.assertEqual(pos_correct_end, result_pos[1], "End year incorrect")

        # Test 3: y = -2x
        x = np.array(range(50))
        y = np.array(range(100,0,-2))
        result_pos = ps5.find_trend(x, y, len(x)//2, True)
        result_neg = ps5.find_trend(x, y, len(x)//2, False)
        neg_correct_start = 0
        neg_correct_end = len(x)//2

        self.assertIsNone(result_pos, "Returned an interval, but should be None.")

        self.assertIsNotNone(result_neg, "Returned None, but valid interval exists.")
        self.assertEqual(neg_correct_start, result_neg[0], "Start year incorrect")
        self.assertEqual(neg_correct_end, result_neg[1], "End year incorrect")

    def test_evaluate_models_on_testing(self):
        x = np.array(range(50))
        y = np.array(range(0,100,2))
        degrees = [1, 2]
        models = ps5.generate_models(x, y, degrees)
        rmse = ps5.evaluate_models_on_testing(x, y, models, False)
        correct_rmse = [0,0]
        self.assertEqual(len(rmse), len(correct_rmse), "RMSE values did not match number of models")
        for index in range(len(correct_rmse)):
            good_enough = math.isclose(correct_rmse[index], rmse[index])
            self.assertTrue(good_enough, "Returned incorrect RMSE values")


if __name__ == '__main__':
    # Run the tests and print verbose output to stderr.
    warnings.simplefilter('ignore', np.RankWarning)
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPS5))
    unittest.TextTestRunner(verbosity=2).run(suite)
