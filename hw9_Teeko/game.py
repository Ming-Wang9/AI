import random
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return False

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        start_time = time.time()
        time_limit = 5  

        piece_count = sum(row.count(self.my_piece) + row.count(self.opp) for row in state)
        drop_phase = piece_count < 8

        if drop_phase:
            empty_spaces = [(i, j) for i in range(5) for j in range(5) if state[i][j] == ' ']
            best_drop = None
            best_score = float('-inf')
        
            for pos in empty_spaces:
                new_state = [row[:] for row in state]
                new_state[pos[0]][pos[1]] = self.my_piece
                score = self.heuristic_game_value(new_state)
                if score > best_score:
                    best_score = score
                    best_drop = pos
            return [best_drop]
    
        best_move = None
        depth = 1
        while time.time() - start_time < time_limit:
            current_best_move = None
            current_best_score = float('-inf')

            moves = []
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                new_i, new_j = i + di, j + dj
                                if 0 <= new_i < 5 and 0 <= new_j < 5 and state[new_i][new_j] == ' ':
                                    moves.append([(new_i, new_j), (i, j)])
        
            moves.sort(key=lambda move: self.heuristic_game_value(self.generate_new_state(state, move)), reverse=True)

            for move in moves:
                if time.time() - start_time > time_limit:
                    break

                new_state = self.generate_new_state(state, move)
                score = self.minimax(new_state, depth, False, float('-inf'), float('inf'), start_time, time_limit)

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
        
            if current_best_move:
                best_move = current_best_move
            depth += 1

        return best_move if best_move else moves[0]
    
    def generate_new_state(self, state, move):
        new_state = [row[:] for row in state]
        new_state[move[1][0]][move[1][1]] = ' '
        new_state[move[0][0]][move[0][1]] = self.my_piece
        return new_state

    def minimax(self, state, depth, is_max, alpha, beta, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return float('inf') if is_max else float('-inf')
            
        game_val = self.game_value(state)
        if game_val != 0:
            return game_val * float('inf')
            
        if depth == 0:
            return self.heuristic_game_value(state)
            
        moves = []
        piece = self.my_piece if is_max else self.opp
        for i in range(5):
            for j in range(5):
                if state[i][j] == piece:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            new_i, new_j = i + di, j + dj
                            if 0 <= new_i < 5 and 0 <= new_j < 5 and state[new_i][new_j] == ' ':
                                moves.append([(new_i, new_j), (i, j)])
        
        if is_max:
            value = float('-inf')
            for move in moves:
                new_state = [row[:] for row in state]
                new_state[move[1][0]][move[1][1]] = ' '
                new_state[move[0][0]][move[0][1]] = piece
                
                value = max(value, self.minimax(new_state, depth-1, False, alpha, beta, start_time, time_limit))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for move in moves:
                new_state = [row[:] for row in state]
                new_state[move[1][0]][move[1][1]] = ' '
                new_state[move[0][0]][move[0][1]] = piece
                
                value = min(value, self.minimax(new_state, depth-1, True, alpha, beta, start_time, time_limit))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def heuristic_game_value(self, state):
        score = 0
        game_val = self.game_value(state)
        if game_val != 0:
            return game_val * float('inf')
        center_weights = [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 9, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1]
        ]
        
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    score += center_weights[i][j]
                elif state[i][j] == self.opp:
                    score -= center_weights[i][j]
        
        for row in range(5):
            for col in range(5):
                if col <= 1:
                    window = [state[row][col+i] for i in range(4)]
                    score += self.evaluate_window(window)
                
                if row <= 1:
                    window = [state[row+i][col] for i in range(4)]
                    score += self.evaluate_window(window)
                
                if row <= 1 and col <= 1:
                    window = [state[row+i][col+i] for i in range(4)]
                    score += self.evaluate_window(window)
                
                if row <= 1 and col >= 3:
                    window = [state[row+i][col-i] for i in range(4)]
                    score += self.evaluate_window(window)
        
        return score

    def evaluate_window(self, window):
        score = 0
        piece_count = window.count(self.my_piece)
        opp_count = window.count(self.opp)
        empty_count = window.count(' ')

        if piece_count == 4:
            score += 1000  
        elif piece_count == 3 and empty_count == 1:
            score += 50
        elif piece_count == 2 and empty_count == 2:
            score += 10

        if opp_count == 3 and empty_count == 1:
            score -= 80  
        elif opp_count == 4:
            score -= 1000  

        return score


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        # TODO: check / diagonal wins
        # TODO: check box wins
        
        # Check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if (state[i][j] != ' ' and
                    state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]):
                    return 1 if state[i][j] == self.my_piece else -1

        # Check / diagonal wins
        for i in range(2):
            for j in range(3, 5):
                if (state[i][j] != ' ' and
                    state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]):
                    return 1 if state[i][j] == self.my_piece else -1

        # Check 2x2 box wins
        for i in range(4):
            for j in range(4):
                pieces = {state[i][j], state[i][j+1], state[i+1][j], state[i+1][j+1]}
                if len(pieces) == 1 and ' ' not in pieces:
                    return 1 if state[i][j] == self.my_piece else -1

  

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()


