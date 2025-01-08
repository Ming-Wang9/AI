import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(9):
        if from_state[i] != 0:
            target = to_state.index(from_state[i])
            distance += abs(i % 3 - target % 3) + abs(i // 3 - target // 3)
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []

    neighbors = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7],
                5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}

    empty_indices = []
    for i, item in enumerate(state):
        if item == 0:
            empty_indices.append(i)

    for empty_index in empty_indices:
        for neighbor_index in neighbors[empty_index]:
            if state[neighbor_index] != 0:
                new_state = state.copy()
                new_state[empty_index], new_state[neighbor_index] = new_state[neighbor_index], new_state[empty_index]
                if new_state not in succ_states:
                    succ_states.append(new_state)
    
    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    pq = []
    visited = []
    h = get_manhattan_distance(state)
    g = 0
    cost = g + h
    parent_index = -1
    max_length = 1
    heapq.heappush(pq, (cost, state, (g, h, parent_index)))
    final_path = []

    while pq:
        current_length = len(pq)
        if current_length > max_length:
            max_length = current_length
        curr = heapq.heappop(pq)
        visited.append(curr)
        current_state = curr[1]
        h = curr[2][1]
        parent_index = curr[2][2]
        g = curr[2][0]

        if current_state == goal_state:
            path = []
            currState = visited[-1]
            while currState[2][2] != -1:
                path.append(currState)
                currState = visited[currState[2][2]]
            path.append(visited[0])
            final_path = path
            break

        parent_index = len(visited) - 1
        g += 1
        for successor in get_succ(current_state):
            h = get_manhattan_distance(successor)
            cost = g + h
            inClosed = False
            inOpen = False
            for item in visited:
                if item[1] == successor:
                    inClosed = True
                    break
            for item in pq:
                if item[1] == successor:
                    inOpen = True
                    if cost < item[0]:
                        heapq.heappush(pq, (cost, successor, (g, h, parent_index)))
                    break
            if not inClosed and not inOpen:
                heapq.heappush(pq, (cost, successor, (g, h, parent_index)))
    moves = 0

    for state in reversed(final_path):
        print(f"{state[1]} h={state[2][1]} moves: {moves}")
        moves += 1
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([2,5,1,4,0,6,7,0,3])
    print()
