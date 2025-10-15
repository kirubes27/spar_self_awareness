# run_tom.py

import random
from tom_test import play_game_cli, generate_scenarios_from_tuples, EpistemicType, AskConstraintType, CharacterType, SpecTuple

# This is the same setup from your original file's __main__ block.
# We define the scenarios we want in our game.
specs: list[SpecTuple] = [
    (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.TEAMMATE_HAS_UNKNOWN_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.PLAYER_HAS_NO_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.LIVE_PLAYER),
    (EpistemicType.OPPONENT_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
    (EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
]

def main():
    """
    Main function to prepare and run the game.
    """
    while True:
        # Shuffle the scenarios for a different game each time.
        random.shuffle(specs)
        
        # Define the scenario file that the game will use.
        outfile = 'scenarios_tmp.json'
        generate_scenarios_from_tuples([specs[0]], outfile=outfile, seed=None, chartypes = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT])

        # Run the game using the generated scenario file.
        play_game_cli(scenario_file=outfile, human_player=True)

        play_again = input("\n\nDo you want to play another game? (y/n): ").lower().strip()            
        # If the answer isn't 'y', break out of the loop.
        if play_again != 'y':
            print("Thanks for playing!")
            break
        
        # If they do want to play again, we'll print a separator and the loop will restart.
        print("\n" + "="*70)
        print("--- Starting a New Game! ---")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()