import random

def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    n = 5
    guess = "R"

    if len(opponent_history) >= n:
        # Pattern recognition logic
        patterns = ["".join(opponent_history[i:i+n]) for i in range(len(opponent_history)-n)]
        last_pattern = "".join(opponent_history[-n:])

        # Count occurrences of each pattern
        pattern_counts = {}
        for pattern in patterns:
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1

        # Find the most common next move after the last pattern
        possible_next_moves = {'R': 0, 'P': 0, 'S': 0}
        for i in range(len(patterns) - 1):
            if patterns[i] == last_pattern:
                next_move = opponent_history[i + n]
                possible_next_moves[next_move] += 1

        # Choose the most likely next move
        most_likely_next_move = max(possible_next_moves, key=possible_next_moves.get)

        # Counter the most likely next move
        if most_likely_next_move == "R":
            guess = "P"
        elif most_likely_next_move == "P":
            guess = "S"
        elif most_likely_next_move == "S":
            guess = "R"

    return guess
