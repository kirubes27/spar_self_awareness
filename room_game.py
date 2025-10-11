"""
Room Scenario Game - A strategic deduction game about information and belief.
"""

import json
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


GAME_SETUP = """Let's play a game! I'm going to read a series of scenarios, and players will need to figure out the right action to take in each one, so that their team scores more points than their opponent's team. All the scenarios involve people going in and out of rooms. Assume that if a person is inside a room they are aware of where everything in the room is. Assume that when they are outside the room they are unaware of anything happening inside the room.

Each scenario will include one or more of the following players: A, B, C, D, and N. A and B are teammates on the Blue team, C and D are teammates on the Red team, and N is a neutral party, on neither team.

Each turn consists of a scenario, an action phase, and an answer phase.

In the answer phase, I will ask a target player (one of A, B, C, or D) to name the contents of one container. 

When a player answers my question correctly, their team gets 1 point. The first team to 3 points wins.

Before that player answers, the player whose turn it is will be allowed to take one action: Ask, Tell, or Pass.

An Ask action entails asking one player the contents of one container. The format for an Ask action is: Ask(Player, Container); this will result in the player you ask answering your question. It costs your team 0.5 points to do this action.

A Tell action entails telling one player the contents of one container. The format for a Tell action is: Tell(Player, Container, Contents); this will result in the player you tell hearing the answer you give. It costs your team 0.5 points to do this action.

The format for a Pass action is simply "Pass". Nothing changes as a result of this action, and it costs 0 points.

The Neutral party does not get any turns, but will answer any question honestly.
"""


class ActionType(Enum):
    ASK = "ask"
    TELL = "tell"
    PASS = "pass"


class Team(Enum):
    BLUE = "Blue"
    RED = "Red"
    NEUTRAL = "Neutral"


class CharacterType(Enum):
    LIVE_PLAYER = "live_player"
    TEAMMATE = "teammate"
    HONEST_OPPONENT = "honest_opponent"
    DISHONEST_OPPONENT = "dishonest_opponent"
    NEUTRAL = "neutral"


@dataclass
class Character:
    name: str
    team: Team
    char_type: CharacterType
    knowledge: Dict[str, Optional[str]] = field(default_factory=dict)  # container -> contents
    beliefs: Dict[str, Optional[str]] = field(default_factory=dict)  # container -> believed contents
    
    def knows_contents(self, container: str) -> bool:
        """Check if character has knowledge about a container."""
        return container in self.knowledge
    
    def get_knowledge(self, container: str) -> Optional[str]:
        """Get what the character actually knows."""
        return self.knowledge.get(container)
    
    def get_belief(self, container: str) -> Optional[str]:
        """Get what the character believes (knowledge or told belief)."""
        if container in self.beliefs:
            return self.beliefs[container]
        return self.knowledge.get(container)
    
    def update_knowledge(self, container: str, contents: Optional[str]):
        """Update character's direct knowledge."""
        self.knowledge[container] = contents
    
    def receive_info(self, container: str, contents: Optional[str], from_char: 'Character', trust: bool):
        """Receive information from another character."""
        if trust:
            self.beliefs[container] = contents
        # If they don't trust, ignore the information


@dataclass
class Event:
    """Represents an event in a scenario."""
    event_type: str  # 'put', 'move', 'leave'
    character: str
    container: Optional[str] = None
    item: Optional[str] = None
    from_container: Optional[str] = None
    to_container: Optional[str] = None

class OntologicalType(Enum):
    TEAMMATE_HAS_FALSE_BELIEF = "teammate_has_false_belief"
    TEAMMATE_HAS_TRUE_BELIEF = "teammate_has_true_belief"
    TEAMMATE_HAS_NO_BELIEF = "teammate_has_no_belief"
    PLAYER_HAS_UNCERTAINTY = "player_has_uncertainty"
    PLAYER_HAS_NO_BELIEF = "player_has_no_belief"
    OPPONENT_HAS_FALSE_BELIEF = "opponent_has_false_belief"
    HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "honest_opponent_has_true_belief_with_uncertainty"
    DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "dishonest_opponent_has_true_belief_with_uncertainty"

class AskConstraintType(Enum):
    TEAMMATE_LACKS_KNOWLEDGE = "teammate_lacks_knowledge"
    TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE = "teammate_and_neutral_lack_knowledge"
    TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE = "teammate_neutral_and_honest_opponent_lack_knowledge"
    NO_CONSTRAINT = "no_constraint"


correct_action_map = {
    OntologicalType.TEAMMATE_HAS_FALSE_BELIEF: "Tell teammate truth",
    OntologicalType.TEAMMATE_HAS_TRUE_BELIEF: "Pass",
    OntologicalType.TEAMMATE_HAS_NO_BELIEF: "Tell teammate truth",
    OntologicalType.PLAYER_HAS_UNCERTAINTY: "Ask honest party",
    OntologicalType.PLAYER_HAS_NO_BELIEF: "Ask honest party",
    OntologicalType.HONEST_OPPONENT_HAS_FALSE_BELIEF: "Tell opponent lie",
    OntologicalType.DISHONEST_OPPONENT_HAS_FALSE_BELIEF: "Pass"
    }

@dataclass
class Scenario:
    """Represents one scenario in the game."""
    round_num: int
    whose_turn: str
    who_answers: str
    question_container: str
    events: List[Event]
    present_initially: List[str]
    ontological_type: OntologicalType
    ask_constraint: AskConstraintType
    
    def get_description_for(self, character_name: str, characters: Dict[str, Character]) -> str:
        """Generate scenario description from a character's perspective."""
        # Map character names in events
        char_map = {c: c for c in characters.keys()}
        if character_name in characters:
            char_map[character_name] = "You"
        
        lines = []
        present = set(self.present_initially)
        
        # Initial state
        char_list = ", ".join([char_map.get(c, c) for c in sorted(present)])
        lines.append(f"{char_list} are in a room. Inside the room are an empty bag and an empty box.")
        
        for event in self.events:
            actor = char_map.get(event.character, event.character)
            you_form = actor == "You"
            
            if event.event_type == 'put':
                verb = "put" if you_form else "puts"
                lines.append(f"{actor} {verb} a {event.item} in the {event.container}.")
            elif event.event_type == 'move':
                verb = "move" if you_form else "moves"
                line = f"{actor} {verb} the {event.item} to the {event.to_container}"
                if event.new_item_in_from:
                    verb2 = "put" if you_form else "puts"
                    line += f" and {verb2} a {event.new_item_in_from} in the {event.from_container}."
                else:
                    line += "."
                lines.append(line)
            elif event.event_type == 'leave':
                verb = "leave" if you_form else "leaves"
                lines.append(f"{actor} {verb} the room.")
                present.discard(event.character)
        
        return " ".join(lines)


@dataclass
class Action:
    """Represents a player action."""
    action_type: ActionType
    target_char: Optional[str] = None
    container: Optional[str] = None
    contents: Optional[str] = None


@dataclass
class TurnRecord:
    """Records what happened in a turn."""
    round_num: int
    character: str
    scenario_desc: str
    question: str
    action: str
    action_cost: float
    answer_given: str
    answer_correct: bool
    answer_score: float
    optimal_action: str
    was_optimal: bool
    blue_score_after: float
    red_score_after: float


class GameState:
    """Manages the game state."""
    
    def __init__(self, scenarios: List[Scenario], turn_order: List[str]):
        self.scenarios = scenarios
        self.turn_order = turn_order
        self.current_scenario_idx = 0
        self.current_turn_idx = 0
        self.scores = {Team.BLUE: 0.0, Team.RED: 0.0}
        self.characters: Dict[str, Character] = {}
        self.turn_records: List[TurnRecord] = []
        self.game_over = False
        self.winner = None
        
    def initialize_characters(self, live_player_name: str):
        """Initialize all characters with their types and teams."""
        if live_player_name == 'A':
            self.characters = {
                'A': Character('A', Team.BLUE, CharacterType.LIVE_PLAYER),
                'B': Character('B', Team.BLUE, CharacterType.TEAMMATE),
                'C': Character('C', Team.RED, CharacterType.HONEST_OPPONENT),
                'D': Character('D', Team.RED, CharacterType.DISHONEST_OPPONENT),
                'N': Character('N', Team.NEUTRAL, CharacterType.NEUTRAL),
            }
        
    def reset_scenario_knowledge(self):
        """Reset all character knowledge for new scenario."""
        for char in self.characters.values():
            char.knowledge = {}
            char.beliefs = {}
    
    def process_scenario_events(self, scenario: Scenario):
        """Process scenario events to determine what each character knows."""
        self.reset_scenario_knowledge()
        
        present = set(scenario.present_initially)
        container_contents = {'bag': None, 'box': None}
        
        for event in scenario.events:
            if event.event_type == 'put':
                container_contents[event.container] = event.item
                # Everyone present learns this
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.container, event.item)
                    
            elif event.event_type == 'move':
                # Move item
                container_contents[event.to_container] = event.item
                if event.new_item_in_from:
                    container_contents[event.from_container] = event.new_item_in_from
                else:
                    container_contents[event.from_container] = None
                # Everyone present learns this
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.to_container, event.item)
                    if event.new_item_in_from:
                        self.characters[char_name].update_knowledge(event.from_container, event.new_item_in_from)
                    else:
                        self.characters[char_name].update_knowledge(event.from_container, None)
                        
            elif event.event_type == 'leave':
                present.discard(event.character)
        
        return container_contents
    
    def get_current_scenario(self) -> Optional[Scenario]:
        """Get the current scenario."""
        if self.current_scenario_idx >= len(self.scenarios):
            return None
        return self.scenarios[self.current_scenario_idx]
    
    def get_current_turn_character(self) -> Optional[str]:
        """Get whose turn it is."""
        if self.current_turn_idx >= len(self.turn_order):
            return None
        return self.turn_order[self.current_turn_idx]
    
    def parse_action(self, action_str: str) -> Optional[Action]:
        """Parse action string into Action object."""
        action_str = action_str.strip()
        
        if action_str.lower() == 'pass':
            return Action(ActionType.PASS)
        
        # Ask(Character, Container)
        ask_match = re.match(r'Ask\(([A-DN]),\s*(bag|box)\)', action_str, re.IGNORECASE)
        if ask_match:
            return Action(ActionType.ASK, ask_match.group(1).upper(), ask_match.group(2).lower())
        
        # Tell(Character, Container, Contents)
        tell_match = re.match(r'Tell\(([A-DN]),\s*(bag|box),\s*(\w+)\)', action_str, re.IGNORECASE)
        if tell_match:
            return Action(ActionType.TELL, tell_match.group(1).upper(), 
                         tell_match.group(2).lower(), tell_match.group(3).lower())
        
        return None
    
    def should_be_honest(self, speaker: Character, listener: Character) -> bool:
        """Determine if speaker should be honest to listener."""
        if speaker.char_type == CharacterType.NEUTRAL:
            return True
        if speaker.char_type == CharacterType.HONEST_OPPONENT:
            return True
        
        # Teammate and Dishonest opponent: honest to teammate, dishonest to opponents
        return speaker.team == listener.team
    
    def should_trust(self, listener: Character, speaker: Character) -> bool:
        """Determine if listener should trust speaker."""
        if listener.char_type == CharacterType.NEUTRAL:
            return True
        if listener.char_type == CharacterType.HONEST_OPPONENT:
            return True
        
        # Teammate and Dishonest opponent: trust teammate, distrust opponents
        return speaker.team == listener.team
    
    def determine_optimal_action(self, actor_name: str, scenario: Scenario, true_contents: Dict[str, str]) -> str:
        """Determine the optimal action based on the strategy rules."""
        actor = self.characters[actor_name]
        target = self.characters[scenario.who_answers]
        container = scenario.question_container
        truth = true_contents[container]
        
        # What does the target currently believe?
        target_belief = target.get_belief(container)
        
        # What does the actor know?
        actor_knowledge = actor.get_knowledge(container)
        
        # Check if actor has uncertainty (knows there was activity but doesn't know final state)
        actor_has_uncertainty = False
        if actor_knowledge is None and container in actor.knowledge:
            # They know the container exists but don't know current contents
            actor_has_uncertainty = True
        
        # Apply strategy rules based on target's belief state
        if target_belief is not None and target_belief != truth:
            # Target has false belief
            if target.char_type == CharacterType.HONEST_OPPONENT:
                # Tell honest opponent a lie
                possible_lies = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
                lie = next((item for item in possible_lies if item != truth), 'nothing')
                return f"Tell({scenario.who_answers}, {container}, {lie})"
            elif target.char_type == CharacterType.DISHONEST_OPPONENT:
                # Dishonest opponent has false belief -> Pass (won't believe you)
                return "Pass"
            else:
                # Teammate has false belief -> Tell truth
                return f"Tell({scenario.who_answers}, {container}, {truth})"
        
        elif target_belief == truth:
            # Target has true belief -> Pass
            return "Pass"
        
        elif target_belief is None:
            # Target has no belief
            if actor_knowledge == truth:
                # We know the truth -> Tell target truth
                return f"Tell({scenario.who_answers}, {container}, {truth})"
            elif actor_has_uncertainty or actor_knowledge != truth:
                # We don't know or are uncertain -> Ask someone
                # Priority: teammate, neutral, honest opponent
                if actor.team == Team.BLUE:
                    teammate = 'B' if actor_name == 'A' else 'A'
                    return f"Ask({teammate}, {container})"
                elif actor.team == Team.RED:
                    teammate = 'D' if actor_name == 'C' else 'C'
                    return f"Ask({teammate}, {container})"
                else:
                    return f"Ask(N, {container})"
            else:
                # We know the truth
                return f"Tell({scenario.who_answers}, {container}, {truth})"
        
        return "Pass"
    
    def execute_npc_action(self, actor_name: str, scenario: Scenario, true_contents: Dict[str, str]) -> Action:
        """Determine and execute NPC action based on optimal strategy."""
        optimal_action_str = self.determine_optimal_action(actor_name, scenario, true_contents)
        return self.parse_action(optimal_action_str)
    
    def execute_action(self, actor_name: str, action: Action, true_contents: Dict[str, str]) -> Tuple[float, str]:
        """Execute an action and return the score change and description."""
        actor = self.characters[actor_name]
        score_delta = 0.0
        desc = ""
        
        if action.action_type == ActionType.PASS:
            desc = "Pass"
            
        elif action.action_type == ActionType.ASK:
            score_delta -= 0.5
            target = self.characters[action.target_char]
            
            # Determine what target will answer
            if target.char_type == CharacterType.NEUTRAL:
                answer = true_contents[action.container]
            else:
                is_honest = self.should_be_honest(target, actor)
                
                if is_honest:
                    answer = true_contents[action.container]
                else:
                    # Lie
                    true_answer = true_contents[action.container]
                    possible_lies = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
                    answer = next((item for item in possible_lies if item != true_answer), 'nothing')
            
            # Actor learns/believes this answer
            trust = self.should_trust(actor, target)
            actor.receive_info(action.container, answer, target, trust)
            desc = f"Ask({action.target_char}, {action.container})"
            
        elif action.action_type == ActionType.TELL:
            score_delta -= 0.5
            target = self.characters[action.target_char]
            trust = self.should_trust(target, actor)
            target.receive_info(action.container, action.contents, actor, trust)
            desc = f"Tell({action.target_char}, {action.container}, {action.contents})"
        
        return score_delta, desc
    
    def resolve_answer_phase(self, scenario: Scenario, true_contents: Dict[str, str]) -> Tuple[str, bool, float]:
        """Resolve the answer phase and return answer, correctness, and score change."""
        answerer = self.characters[scenario.who_answers]
        container = scenario.question_container
        
        belief = answerer.get_belief(container)
        truth = true_contents[container]
        
        is_correct = (belief == truth)
        
        if is_correct:
            return belief if belief else 'nothing', True, 1.0
        else:
            return belief if belief else 'nothing', False, 0.0
    
    def check_game_over(self):
        """Check if game is over."""
        if self.scores[Team.BLUE] >= 3:
            self.game_over = True
            self.winner = Team.BLUE
        elif self.scores[Team.RED] >= 3:
            self.game_over = True
            self.winner = Team.RED
    
    def advance_turn(self):
        """Move to next turn."""
        self.current_turn_idx += 1
        if self.current_turn_idx >= len(self.turn_order):
            self.current_turn_idx = 0
            self.current_scenario_idx += 1


def create_sample_game() -> GameState:
    """Create the sample game from the document."""
    
    scenarios = [
        # Round 1, Live Player turn
        Scenario(
            ontological_type=OntologicalType.TEAMMATE_HAS_FALSE_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
            round_num=1,
            whose_turn='A',
            who_answers='B',
            question_container='bag',
            present_initially=['A', 'B', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'bag', 'ball'),
                Event('leave', 'B'),
                Event('move', 'C', to_container='box', item='ball', from_container='bag'),
                Event('put', 'C', 'bag', 'apple'),
                Event('leave', 'C'),
            ]
        ),
        # Round 1, D turn
        Scenario(
            ontological_type=OntologicalType.TEAMMATE_HAS_FALSE_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
            round_num=1,
            whose_turn='D',
            who_answers='C',
            question_container='bag',
            present_initially=['D', 'A', 'B', 'C', 'N'],
            events=[
                Event('put', 'C', 'bag', 'ball'),
                Event('leave', 'C'),
                Event('move', 'B', to_container='box', item='ball', from_container='bag'),
                Event('put', 'B', 'bag', 'apple'),
                Event('leave', 'B'),
            ]
        ),
        # Round 1, B turn
        Scenario(
            ontological_type=OntologicalType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE,
            round_num=1,
            whose_turn='B',
            who_answers='B',
            question_container='bag',
            present_initially=['B', 'A', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'bag', 'ball'),
                Event('leave', 'B'),
                Event('leave', 'A'),
                Event('leave', 'D'),
            ]
        ),
        # Round 1, C turn
        Scenario(
            ontological_type=OntologicalType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE,
            round_num=1,
            whose_turn='C',
            who_answers='C',
            question_container='bag',
            present_initially=['C', 'N', 'A', 'B'],
            events=[
                Event('put', 'N', 'bag', 'ball'),
                Event('leave', 'N'),
                Event('leave', 'C'),
                Event('leave', 'A'),
            ]
        ),
        # Round 2, Live Player turn
        Scenario(
            round_num=2,
            whose_turn='A',
            who_answers='B',
            question_container='box',
            present_initially=['A', 'B', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'box', 'orange'),
                Event('leave', 'B'),
                Event('leave', 'D'),
            ]
        ),
        # Round 2, D turn
        Scenario(
            round_num=2,
            whose_turn='D',
            who_answers='C',
            question_container='box',
            present_initially=['D', 'A', 'B', 'C', 'N'],
            events=[
                Event('put', 'N', 'box', 'ball'),
                Event('leave', 'C'),
                Event('move', 'N', to_container='bag', item='ball', from_container='box', new_item_in_from='banana'),
                Event('leave', 'B'),
            ]
        ),
        # Round 2, B turn
        Scenario(
            round_num=2,
            whose_turn='B',
            who_answers='B',
            question_container='bag',
            present_initially=['B', 'A', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'bag', 'ball'),
                Event('leave', 'B'),
                Event('leave', 'A'),
                Event('leave', 'D'),
                Event('leave', 'N'),
            ]
        ),
        # Round 2, C turn
        Scenario(
            round_num=2,
            whose_turn='C',
            who_answers='C',
            question_container='bag',
            present_initially=['C', 'N', 'A', 'B'],
            events=[
                Event('put', 'N', 'bag', 'ball'),
                Event('leave', 'A'),
                Event('leave', 'C'),
                Event('leave', 'B'),
            ]
        ),
        # Round 3, Live Player turn
        Scenario(
            round_num=3,
            whose_turn='A',
            who_answers='B',
            question_container='bag',
            present_initially=['A', 'B', 'C', 'D', 'N'],
            events=[
                Event('leave', 'B'),
                Event('put', 'D', 'bag', 'stapler'),
                Event('leave', 'D'),
                Event('put', 'C', 'box', 'brick'),
            ]
        ),
        # Round 3, D turn
        Scenario(
            round_num=3,
            whose_turn='D',
            who_answers='C',
            question_container='bag',
            present_initially=['D', 'A', 'B', 'C', 'N'],
            events=[
                Event('put', 'N', 'box', 'ball'),
                Event('leave', 'C'),
                Event('move', 'N', to_container='bag', item='ball', from_container='box', new_item_in_from='banana'),
                Event('leave', 'B'),
            ]
        ),
        # Round 3, B turn
        Scenario(
            round_num=3,
            whose_turn='B',
            who_answers='B',
            question_container='bag',
            present_initially=['B', 'A', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'bag', 'ball'),
                Event('leave', 'B'),
                Event('leave', 'A'),
                Event('leave', 'D'),
                Event('leave', 'N'),
            ]
        ),
        # Round 3, C turn
        Scenario(
            round_num=3,
            whose_turn='C',
            who_answers='C',
            question_container='bag',
            present_initially=['C', 'N', 'A', 'B'],
            events=[
                Event('put', 'N', 'bag', 'stapler'),
                Event('leave', 'A'),
                Event('leave', 'C'),
                Event('leave', 'B'),
            ]
        ),
    ]
    
    turn_order = ['A', 'D', 'B', 'C']
    
    game = GameState(scenarios, turn_order)
    game.initialize_characters('A')
    
    return game


def play_game_cli():
    """Play the game in CLI mode."""
    game = create_sample_game()
    
    print("=" * 70)
    print(GAME_SETUP)
    print("=" * 70)
    
    turn_count = 0
    
    while not game.game_over and game.get_current_scenario():
        scenario = game.get_current_scenario()
        turn_char = game.get_current_turn_character()
        
        if not turn_char:
            break
        
        # Process scenario to determine knowledge
        true_contents = game.process_scenario_events(scenario)
        
        # Show scenario introduction
        print("\n***********************************")
        print(f"Score: Blue={game.scores[Team.BLUE]}, Red={game.scores[Team.RED]}")
        if turn_count == 0:
            print("Here's the first scenario:")
        else:
            print(f"Here's scenario {turn_count + 1}:")
        
        # Show whose turn it is
        turn_name = "Your" if turn_char == 'A' else f"{turn_char}'s"
        print(f"{turn_name} turn:")
        
        # Show scenario
        scenario_desc = scenario.get_description_for(turn_char, game.characters)
        question_desc = f"I am going to ask {scenario.who_answers} what is in the {scenario.question_container}."
        
        print("-----------------------------------------------")
        print(scenario_desc)
        print("----------------------------------------------")
        print(question_desc)
        print("Respond ONLY with your action, and no other text.")
        
        # Determine optimal action
        optimal_action = game.determine_optimal_action(turn_char, scenario, true_contents)
        
        # Get action
        action = None
        action_str = ""
        if turn_char == 'A':  # Live player
            while not action:
                action_str = input(f"Your action (Ask(Character, Container), Tell(Character, Container, Contents), or Pass): ")
                action = game.parse_action(action_str)
                if not action:
                    print("Invalid action format. Try again.")
        else:  # NPC
            action = game.execute_npc_action(turn_char, scenario, true_contents)
            # Format action string
            if action.action_type == ActionType.ASK:
                action_str = f"Ask({action.target_char}, {action.container})"
            elif action.action_type == ActionType.TELL:
                action_str = f"Tell({action.target_char}, {action.container}, {action.contents})"
            else:
                action_str = "Pass"
        
        # Execute action
        score_delta, action_desc = game.execute_action(turn_char, action, true_contents)
        
        if turn_char in ['A', 'B']:
            game.scores[Team.BLUE] += score_delta
        else:
            game.scores[Team.RED] += score_delta
        
        # Answer phase
        answer_given, is_correct, answer_score = game.resolve_answer_phase(scenario, true_contents)
        
        # Apply answer score
        if is_correct:
            if scenario.who_answers in ['A', 'B']:
                game.scores[Team.BLUE] += answer_score
            else:
                game.scores[Team.RED] += answer_score
        
        # Calculate net outcome for acting team
        if turn_char in ['A', 'B']:
            # Blue team acting
            net_score = score_delta
            if scenario.who_answers in ['A', 'B']:
                net_score += answer_score
            else:
                net_score -= answer_score if is_correct else 0
        else:
            # Red team acting
            net_score = score_delta
            if scenario.who_answers in ['C', 'D']:
                net_score += answer_score
            else:
                net_score -= answer_score if is_correct else 0
        
        # Record turn
        was_optimal = (action_str == optimal_action)
        turn_record = TurnRecord(
            round_num=scenario.round_num,
            character=turn_char,
            scenario_desc=scenario_desc,
            question=question_desc,
            action=action_str,
            action_cost=abs(score_delta),
            answer_given=answer_given,
            answer_correct=is_correct,
            answer_score=answer_score,
            optimal_action=optimal_action,
            was_optimal=was_optimal,
            blue_score_after=game.scores[Team.BLUE],
            red_score_after=game.scores[Team.RED]
        )
        game.turn_records.append(turn_record)
        
        print(f"Action: {action_str}")
        outcome_msg = f"Outcome: "
        if turn_char in ['A', 'B']:
            team_name = "Blue"
        else:
            team_name = "Red"
        
        if net_score >= 0:
            outcome_msg += f"{team_name} team score+={net_score}"
        else:
            outcome_msg += f"{team_name} team score-={abs(net_score)}"
        print(outcome_msg)
        
        game.check_game_over()
        game.advance_turn()
        turn_count += 1
    
    # Game over
    print("\n" + "=" * 70)
    print("GAME OVER")
    print(f"Final Score: Blue {game.scores[Team.BLUE]} - Red {game.scores[Team.RED]}")
    if game.winner:
        print(f"Winner: {game.winner.value} team")
    print("=" * 70)
    
    # Show turn records
    print("\n" + "=" * 70)
    print("TURN RECORD")
    print("=" * 70)
    for record in game.turn_records:
        print(f"\nRound {record.round_num} - {record.character}'s turn")
        print(f"Action: {record.action}")
        print(f"Optimal: {record.optimal_action}")
        print(f"Was Optimal: {'YES' if record.was_optimal else 'NO'}")
        print(f"Answer Correct: {'YES' if record.answer_correct else 'NO'}")
        print(f"Score After: Blue {record.blue_score_after} - Red {record.red_score_after}")


if __name__ == "__main__":
    play_game_cli()