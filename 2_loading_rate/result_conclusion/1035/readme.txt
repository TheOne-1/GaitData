layer_to_explore = [
    [],                             # 0 hidden layer
    [50],
    [50, 50],
    [50, 50, 50],
    [50, 50, 50, 50, 50],
    [50 for _ in range(10)],
    [50 for _ in range(20)],
    [50 for _ in range(50)],
    [50 for _ in range(200)],

    [2, 2, 2],
    [3, 3, 3],
    [5, 5, 5],
    [10, 10, 10],
    [20, 20, 20],
    [50, 50, 50],
    [200, 200, 200],
    [500, 500, 500],
    [1000, 1000, 1000],
]
layer_to_explore = [
    [50, 50, 50],
    [50 for _ in range(35)],
    [50 for _ in range(75)],
    [50 for _ in range(100)],
]