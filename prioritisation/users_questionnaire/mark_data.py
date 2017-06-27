from process_questionnaire_data import jsonfy_questionnaire_data,  separate_spend_data

mark_data = [[
    [1500.00, 5, 4435, 119, 10.71, 35.00, 1],
    [450.00, 5, 22, 76, 284.25, 1.31, 4],
    [38.00, 5, 42, 33, 1.49, 0.36, 5],
    [115000.00, 5, 20, 75, 1233.87, 78.12, 2],
    [5096.00, 5, 181, 103, 2.77, 1.31, 3]
],
    [
    [10000.00, 5, 21, 57, 143.09, 162.76, 2],
    [5096.00, 5, 181, 103, 2.77, 1.31, 3],
    [1000.00, 25, 64, 105, 3.13, 2.00, 5],
    [3500.00, 25, 47, 90, 0.68, 2.00, 4],
    [80000.00, 5, 61, 83, 477.54, 162.76, 1]
],
    [
    [600.00, 3, 13, 49, 243.40, 40.54, 2],
    [1680.00, 286, 61, 84, 390.38, 125.00, 4],
    [75.00, 10, 31, 75, 23.57, 1.31, 3],
    [23575.00, 19, 2, 16, 350.24, 68.00, 1],
    [300.00, 23, 45, 89, 221.27, 111.38, 5]
],
    [
    [2900.00, 318, 24, 83, 9.83, 1.31, 4],
    [8750.00, 11, 52, 95, 493.86, 121.36, 2],
    [100.00, 13, 38, 62, 25.96, 1.31, 5],
    [25000.00, 3, 31, 82, 4.78, 22.40, 1],
    [8910.00, 5, 50, 72, 39.58, 125.00, 3]
],
    [
    [2500.00, 28, 2949, 792, 28.59, 68.00, 1],
    [8000.00, 89, 22, 25, None, None,  5],
    [110000.00, 28, 98, 132,  1757.42, 54.79, 4],
    [2000.00, 28, 650, 440, 85.07, 54.79, 2],
    [1200.00, 28, 146, 163, 189.04, 111.38, 3]
],
    [
    [288.00, 28, 65, 26, None, None,  5],
    [3700.00, 7, 1235, 205, 873.50, 36.89, 1],
    [4000.00, 28, 342, 214, None, None,  3],
    [10000.00, 28, 66, 77, 747.60, 54.79, 4],
    [2000.00, 28, 650, 440, 85.07, 54.79, 2]
],
    [
    [3500.00, 17, 46, 89, 0.45, 2.00, 4],
    [1995.00, 28, 851, 471, 227.32, 68.00, 1],
    [2000.00, 7, 34, 39, 355.93, 36.90, 3],
    [65000.00, 17, 8, 26, None, None,  2],
    [2000.00, 28, 126, 102, 8.25, 125.00, 5]
],
    [
    [1750.00, 28, 44, 59, 19.89, 111.38, 5],
    [8000.00, 89, 22, 25, None, None,  3],
    [2500.00, 28, 2949, 792, 28.59, 68.00, 1],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 2],
    [38.00, 28, 19, 48, None, None,  4]
],
    [
    [110000.00, 28, 98, 132,  1757.42, 54.79, 5],
    [9174.00, 28, 959, 537, 20.37, 5.44, 1],
    [1000.00, 28, 567, 467, None, None,  2],
    [2000.00, 32, 534, 44, 34.53, 1.32, 3],
    [4266.00, 28, 165, 73, None, None,  4]
],
    [
    [10769.00, 28, 34, 41, 4.95, 5.44, 3],
    [2200.00, 7, 21, 32, 164.43, 36.90, 4],
    [5499.00, 28, 162, 159, 13.02, 1.31, 1],
    [379.00, 28, 76, 207, None, None,  5],
    [8000.00, 89, 22, 25, None, None,  2]
],
    [
    [3000.00, 28, 110, 117, 170.40, 54.79, 5],
    [250.00, 28, 14, 13, None, None,  3],
    [425.00, 28, 47, 44, None, None,  4],
    [575.00, 28, 165, 129, None, None,  2],
    [1500.00, 28, 160, 169, 6.30, 35.00, 1]
],
    [
    [1590.00, 28, 165, 171, 13.18, 1.31, 1],
    [25000.00, 28, 21, 30, 1.34, 22.40, 3],
    [5750.00, 28, 110, 123, 227.91, 111.37, 5],
    [115.00, 28, 128, 142, 7.88, 1.31, 2],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 4]
],
    [
    [8000.00, 28, 109, 90, None, None,  5],
    [115.00, 28, 128, 142, 7.88, 1.31, 3],
    [15000.00, 28, 209, 236, 51.22, 113.04, 1],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 4],
    [588.00, 28, 151, 209, 59.39, 1.31, 2]
],
    [
    [10769.00, 28, 34, 41, 4.95, 5.44, 3],
    [5096.00, 28, 144, 156, 2.27, 1.31, 1],
    [6000.00, 28, 75, 83, 96.55, 137.35, 4],
    [2000.00, 28, 126, 102, 8.25, 125.00, 2],
    [250.00, 28, 14, 13, None, None,  5]
],
    [
    [10500.00, 28, 149, 149, 44.64, 22.40, 3],
    [1750.00, 28, 44, 59, 19.89, 111.38, 4],
    [250.00, 25, 92, 125, None, None,  5],
    [6000.00, 67, 678, 189, None, None,  1],
    [425.00, 28, 163, 181, None, None,  2]
],
    [
    [12963.00, 23, 226, 143, 7.25, 4.02, 1],
    [2000.00, 34, 182, 109, 34.55, 1.32, 4],
    [2250.00, 23, 193, 113, 40.90, 1.31, 2],
    [2250.00, 23, 193, 113, 40.90, 1.31, 2],
    [2000.00, 23, 150, 89, 9.37, 125.00, 5]
],
    [
    [3500.00, 23, 73, 78, 176.48, 40.54, 4],
    [70000.00, 23, 43, 62, 801.34, 54.79, 2],
    [12000.00, 23, 79, 73, 49.14, 113.04, 3],
    [25000.00, 23, 179, 51, 2.85, 22.40, 1],
    [3423.00, 23, 107, 87, 12.14, 1.31, 5]
],
    [
    [250.00, 23, 27, 33, None, None,  3],
    [200.00, 23, 60, 88, None, None,  5],
    [150.00, 23, 130, 111, None, None,  1],
    [250.00, 23, 12, 12, None, None,  2],
    [800.00, 23, 43, 36, None, None,  4]
],
    [
    [5000.00, 16, 148, 95,  2296.18, 59.16, 3],
    [115.00, 16, 197, 89, 9.88, 1.31, 2],
    [5000.00, 16, 104, 106, 85.70, 35.00, 5],
    [5096.00, 16, 205, 119, 2.87, 1.31, 1],
    [1450.00, 16, 71, 90, 157.85, 125.00, 4]
],
    [
    [8750.00, 16, 100, 83, 147.06, 121.36, 5],
    [588.00, 16, 104, 104, 37.15, 1.31, 4],
    [6000.00, 27, 111, 94, 2.00, 1.32, 2],
    [60000.00, 16, 96, 89, 306.77, 113.04, 3],
    [1750.00, 16, 38, 49, 19.82, 111.38, 1]
],
    [
    [1800.00, 16, 82, 90, None, None,  4],
    [2000.00, 16, 46, 74, None, None,  3],
    [200.00, 16, 83, 103, None, None,  5],
    [425.00, 16, 150, 120, None, None,  1],
    [253.00, 16, 10, 36, None, None,  2]
],
    [
    [2500.00, 28, 2949, 792, 28.59, 68.00, 1],
    [8000.00, 89, 22, 25, None, None,  4],
    [110000.00, 28, 98, 132,  1757.42, 54.79, 5],
    [2000.00, 28, 650, 440, 85.07, 54.79, 2],
    [1200.00, 28, 146, 163, 189.04, 111.38, 3]
],
    [
    [70000.00, 16, 58, 64, 743.35, 59.16, 1],
    [8500.00, 16, 120, 112, 100.00, 59.16, 3],
    [2500.00, 16, 50, 60, 21.30, 35.00, 5],
    [1500.00, 16, 310, 144, 22.30, 35.00, 2],
    [10000.00, 16, 18, 44, 52.25, 125.00, 4]
],
    [
    [10000.00, 16, 39, 56, 247.40, 59.16, 2],
    [1450.00, 16, 89, 91, 320.97, 125.00, 5],
    [4500.00, 27, 57, 69, 0.59, 1.32, 3],
    [2000.00, 27, 255, 131, 3.30, 1.32, 1],
    [150.00, 16, 106, 115, 83.77, 125.00, 4]
],
    [
    [758.00, 16, 110, 114, None, None,  3],
    [3930.00, 55, 12, 32, None, None,  1],
    [1500.00, 77, 33, 31, None, None,  2],
    [425.00, 16, 110, 99, None, None,  4],
    [4000.00, 16, 100, 177, None, None,  5]
],
    [
    [20000.00, 16, 20, 20, 20, 20, 1],
    [1450.00, 16, 20, 20, 30, 30, 2],
    [1450.00, 25, 20, 20, 40, 40, 3],
    [1450.00, 16, 71, 90, 125.00, 100.0, 4],
    [1450.00, 16, 71, 90, 100, 100.00, 5],
    [1450.00, 16, 90, 80, 30, 30, 6],
    [1450.00, 5, 110, 100, 10, 10, 7],
    [1450.00, 5, 100, 100, 11, 10, 8],
    [1450.00, 50, 100, 100, 5, 5, 10],
    [1450.00, 1, 100, 100, 5, 5, 9]
    ]
]

mark_regular_data,  mark_spend_data = separate_spend_data( mark_data)

mark_regular_json_data = jsonfy_questionnaire_data( mark_regular_data)
mark_spend_json_data = jsonfy_questionnaire_data( mark_spend_data)

mark_json_data = jsonfy_questionnaire_data( mark_data)