import heapq

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(len(from_state)):
        if from_state[i] == 0:
            continue
        goal_index = to_state.index(from_state[i])
        current_row, current_col = divmod(i, 3)
        goal_row, goal_col = divmod(goal_index, 3)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)

    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state,goal_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    successors = []
    neighbors = {
        0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
        3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
        6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
    }
    empty_positions = [i for i, tile in enumerate(state) if tile == 0]
    for empty_pos in empty_positions:
        for neighbor_pos in neighbors[empty_pos]:
            if state[neighbor_pos] != 0:  
                new_state = list(state)
                new_state[empty_pos], new_state[neighbor_pos] = new_state[neighbor_pos], new_state[empty_pos]
                successors.append(new_state)
    
    return sorted(successors)

def get_inversion_count(state):
    """Helper function to count inversions in the puzzle state."""
    non_zero = [n for n in state if n != 0]  
    inversions = sum(1 for i, num1 in enumerate(non_zero)
                    for num2 in non_zero[i + 1:] if num1 > num2)
    return inversions

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # This is a format helper，which is only designed for format purpose.
    # define "solvable_condition" to check if the puzzle is really solvable
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute "max_length", it might be useful in debugging
    # it can help to avoid any potential format issue.

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)
    num_tiles = sum(1 for x in state if x != 0)
    
    inversions = get_inversion_count(state)
    if num_tiles % 2 == 0:
        is_solvable = inversions % 2 == 0
    else:
        is_solvable = True
    if not is_solvable:
        print(False)
        return
    print(True)
    
    pq = []  
    visited = {}  
    max_queue_length = 1
    initial_h = get_manhattan_distance(state, goal_state)
    initial_g = 0
    initial_f = initial_g + initial_h
    
    state_tuple = tuple(state)
    heapq.heappush(pq, (initial_f, state_tuple, initial_g, None))
    visited[state_tuple] = (None, initial_g, initial_h)
    
    while pq:
        max_queue_length = max(max_queue_length, len(pq))
        _, current_state_tuple, g_score, _ = heapq.heappop(pq)
        current_state = list(current_state_tuple)
        if current_state == list(goal_state):
            path = []
            temp_state = current_state_tuple
            while temp_state is not None:
                parent, g, h = visited[temp_state]
                path.append((list(temp_state), h, g))
                temp_state = parent
            for state_info in reversed(path):
                state, h, moves = state_info
                print(state, f"h={h}", f"moves: {moves}")

            print(f"Max queue length: {max_queue_length}")
            return
            
        for succ_state in get_succ(current_state):
            succ_tuple = tuple(succ_state)
            new_g_score = g_score + 1
            new_h_score = get_manhattan_distance(succ_state, goal_state)
            new_f_score = new_g_score + new_h_score
            
            if succ_tuple not in visited or new_g_score < visited[succ_tuple][1]:
                visited[succ_tuple] = (current_state_tuple, new_g_score, new_h_score)
                heapq.heappush(pq, (new_f_score, succ_tuple, new_g_score, current_state_tuple))

                
if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    solve([2,5,1,4,0,6,7,0,3])
    print()

    solve([4,3,0,5,1,6,7,2,0])
    print()
