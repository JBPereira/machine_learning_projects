from process_questionnaire_data import jsonfy_questionnaire_data, separate_spend_data

abigail_training_set = [[[1500.00, 5, 4435, 119, 10.71, 35.00, 5],
    [450.00, 5, 22, 76, 284.25, 1.31, 1],
    [38.00, 5, 42, 33, 1.49, 0.36, 3],
    [115000.00, 5, 20, 75, 1233.87, 78.12, 2],
    [5096.00, 5, 181, 103, 2.77, 1.31, 4]
    ],
[
    [10000.00, 5, 21, 57, 143.09, 162.76, 5],
    [5096.00, 5, 181, 103, 2.77, 1.31, 2],
    [1000.00, 25, 64, 105, 3.13, 2.00, 3],
    [3500.00, 25, 47, 90, 0.68, 2.00, 4],
    [80000.00, 5, 61, 83, 477.54, 162.76, 1]
    ],
[
    [7000.00, 5, 29, 63,  1223.04, 137.35, 4],
    [10000.00, 5, 32, 78,  296.19, 78.12, 3],
    [500.00, 5, 47, 68, 13.54, 40.54, 5],
    [3500.00, 5, 287, 113, 64.67, 78.12, 1],
    [31625.00, 5, 33, 68,  429.63, 68.00, 2]
    ],
[
    [8910.00, 5, 50, 72, 39.58, 125.00, 5],
    [5475.00, 5, 134, 101, 7.13, 1.31, 1],
    [5000.00, 5, 19, 50, 79.84, 40.54, 4],
    [1690.00, 5, 52, 80,  376.36,  125.00, 2],
    [130000.00, 5, 35, 71,  691.76, 78.12, 3]
    ],
[
    [600.00, 3, 13, 49,  243.40, 40.54, 1],
    [1680.00, 286, 61, 84,  390.38, 125.00, 5],
    [75.00, 10, 31, 75, 23.57, 1.31, 2],
    [23575.00, 19, 2, 16,  350.24, 68.00, 3],
    [300.00, 23, 45, 89,  221.27, 111.38, 4]
    ],
[
    [2900.00, 318, 24, 83, 9.83, 1.31, 3],
    [8750.00, 11, 52, 95,  493.86,  121.36, 2],
    [100.00, 13, 38, 62, 25.96, 1.31, 1],
    [25000.00, 3, 31, 82, 4.78, 22.40, 4],
    [8910.00, 5, 50, 72, 39.58, 125.00, 5]
    ],
[
    [2500.00, 28, 2949, 792, 28.59, 68.00, 1],
    [8000.00, 89, 22, 25,0,  None,  5],
    [110000.00, 28, 98, 132,  1757.42,  54.79, 4],
    [2000.00, 28, 650, 440, 85.07, 54.79, 2],
    [1200.00, 28, 146, 163,  189.04, 111.38, 3]
    ],
[
    [288.00, 28, 65, 26, 0, None, 5],
    [3700.00, 7, 1235, 205,  873.50, 36.89, 1],
    [4000.00, 28, 342, 214, 0, None, 3],
    [10000.00, 28, 66, 77,  747.60, 54.79, 4],
    [2000.00, 28, 650, 440, 85.07, 54.79, 2]
    ],
[
    [3500.00, 17, 46, 89, 0.45, 2.00, 5],
    [1995.00, 28, 851, 471,  227.32, 68.00, 1],
    [2000.00, 7, 34, 39,  355.93, 36.90, 4],
    [65000.00, 17, 8, 26, 0, None, 3],
    [2000.00, 28, 126, 102, 8.25, 125.00, 2]
    ],
[
    [1750.00, 28, 44, 59, 19.89, 111.38, 5],
    [8000.00, 89, 22, 25, 0, None, 4],
    [2500.00, 28, 2949, 792, 28.59, 68.00, 1],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 2],
    [38.00, 28, 19, 48, 0, None, 3]
    ],
[
    [110000.00, 28, 98, 132,  1757.42, 54.79, 5],
    [9174.00, 28, 959, 537, 20.37, 5.44, 1],
    [1000.00, 28, 567, 467, 0, None, 2],
    [2000.00, 32, 534, 44, 34.53, 1.32, 3],
    [4266.00, 28, 165, 73, 0, None, 4]
    ],
[
    [10769.00, 28, 34, 41, 4.95, 5.44, 4],
    [2200.00, 7, 21, 32,  164.43, 36.90, 2],
    [5499.00, 28, 162, 159, 13.02, 1.31, 1],
    [379.00, 28, 76, 207, 0, None, 5],
    [8000.00, 89, 22, 25, 0, None, 3]
    ],
[
    [3000.00, 28, 110, 117,  170.40, 54.79, 3],
    [250.00, 28, 14, 13, 0, None, 4],
    [425.00, 28, 47, 44, 0, None, 5],
    [575.00, 28, 165, 129, 0, None, 1],
    [1500.00, 28, 160, 169, 6.30, 35.00, 2]
    ],
[
    [1590.00, 28, 165, 171, 13.18, 1.31, 1],
    [25000.00, 28, 21, 30, 1.34, 22.40, 5],
    [5750.00, 28, 110, 123,  227.91, 111.37, 4],
    [115.00, 28, 128, 142, 7.88, 1.31, 2],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 3]
	],
[
    [8000.00, 28, 109, 90, 0, None, 5], 
    [115.00, 28, 128, 142, 7.88, 1.31, 3],
    [15000.00, 28, 209, 236, 51.22, 113.04, 2],
    [70000.00, 28, 66, 83,  1399.62, 54.79, 4],
    [588.00, 28, 151, 209, 59.39, 1.31, 1]
	],
[
    [10769.00, 28, 34, 41, 4.95, 5.44, 4],
    [5096.00, 28, 144, 156, 2.27, 1.31, 1],
    [6000.00, 28, 75, 83, 96.55, 137.35, 5],
    [2000.00, 28, 126, 102, 8.25, 125.00, 2],
    [250.00, 28, 14, 13, 0, None, 3]
	],
[
    [10500.00, 28, 149, 149, 44.64, 22.40, 3],
    [1750.00, 28, 44, 59, 19.89, 111.38, 4],
    [250.00, 25, 92, 125, 0, None, 5],
    [6000.00, 67, 678, 189, 0, None, 1],
    [425.00, 28, 163, 181, 0, None, 2]
	],
[
    [12963.00, 23, 226, 143, 7.25, 4.02, 1],
    [2000.00, 34, 182, 109, 34.55, 1.32, 4],
    [2250.00, 23, 193, 113, 40.90, 1.31, 3],
    [2250.00, 23, 193, 113, 40.90, 1.31, 2],
    [2000.00, 23, 150, 89, 9.37, 125.00, 5]
    ],
[
    [3500.00, 23, 73, 78, 176.48, 40.54, 3],
    [70000.00, 23, 43, 62, 801.34, 54.79, 5],
    [12000.00, 23, 79, 73, 49.14, 113.04, 4],
    [25000.00, 23, 179, 51, 2.85, 22.40, 1],
    [3423.00, 23, 107, 87, 12.14, 1.31, 2]
    ],
[
    [250.00, 23, 27, 33, None, None, 3],
    [200.00, 23, 60, 88, None, None, 5],
    [150.00, 23, 130, 111, None, None, 1],
    [250.00, 23, 12, 12, None, None, 2],
    [800.00, 23, 43, 36, None, None, 4]
    ],
[
    [8000.00, 24, 34.27, 73.41, 16.31, 113.04, 3],
    [7000.00, 27, 27.92, 44.83, 104.61, 113.04, 4],
    [7000.00, 23, 97.8, 82.31, 35.56, 113.04, 5],
    [65000.00, 24, 61.33, 79.52, 957.13, 113.04, 1],
    [200, 29, 51.91, 55.26, 18.42, 137.35, 2]
    ],
[
    [3500.00, 26, 66.58, 68.73, 21.1, 137.35, 2],
    [250, 22, 39.93, 45.58, 9.49, 29.26, 3],
    [500, 27, 41.5, 47.69, 7.95, 29.26, 4],
    [2000.00, 29, 170.57, 130, 433.33, 29.26, 1],
    [1000.00, 27, 59.31, 96.97, 23.09, 101, 5]
    ],
[
    [1400.00, 27, 69.67, 73.45, 171.39, 101, 3],
    [2000.00, 25, 63.01, 58.75, 11.52, 101, 5],
    [8000.00, 27, 19.52, 15.72, 29.95, 101, 4],
    [7500.00, 20, 113.7, 109.92, 196.28, 98.06, 2],
    [50000.00, 28, 135.17, 135.81, 419.17, 101, 1]
    ],
[
    [250, 26, 107.71, 95.86, 39.94, 101, 4],
    [12250.00, 27, 86.3, 85.62, 1.58, 3.58, 5],
    [3000.00, 20, 151.28, 139.67, 12.93, 3.58, 2],
    [2000.00, 29, 469.86, 278.99, 33.21, 3.58, 1],
    [20000.00, 27, 123.42, 119.69, 4.32, 3.58, 3]
    ],
[
    [3000.00, 29, 75.21, 80.3, 20.08, 62, 5],
    [23025.00, 22, 22.17, 19.71, 75.64, 62, 4],
    [3000.00, 21, 521.14, 399.1, 46.41, 62, 1],
    [30680.00, 28, 43.19, 45.54, 388.09, 62, 3],
    [2295.00, 28, 365.06, 206.76, 790.85, 62, 2]
    ]
]


abigail_regular_data, abigail_spend_data = separate_spend_data(abigail_training_set)

ab_regular_json_data = jsonfy_questionnaire_data(abigail_regular_data)
ab_spend_json_data = jsonfy_questionnaire_data(abigail_spend_data)

ab_json_data = jsonfy_questionnaire_data(abigail_training_set)




