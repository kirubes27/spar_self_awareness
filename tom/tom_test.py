"""
Room Scenario Game - A strategic deduction game about information and belief.
"""

import json
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

WINNING_SCORE = 5

GAME_SETUP = """Let's play a game! I'm going to read a series of scenarios, and players will need to figure out the right action to take in each one, so that their team scores more points than their opponent's team. All the scenarios involve people going in and out of rooms. Assume that if a person is inside a room they are aware of where everything in the room is. Assume that when they are outside the room they are unaware of anything happening inside the room.

Each scenario will include one or more of the following players: A, B, C, D, and N. A and B are teammates on the Blue team, C and D are teammates on the Red team, and N is a neutral party, on neither team.

Each turn consists of a scenario, an action phase, and an answer phase.

In the answer phase, I will ask a target player (one of A, B, C, or D) to name the contents of one container. 

When a player answers my question correctly, their team gets 1 point. The first team to {WINNING_SCORE} points wins.

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
    HONEST_TEAMMATE = "honest_teammate"
    DISHONEST_TEAMMATE = "dishonest_teammate"
    HONEST_OPPONENT = "honest_opponent"
    DISHONEST_OPPONENT = "dishonest_opponent"
    NEUTRAL = "neutral"


class EpistemicType(Enum):
    TEAMMATE_HAS_FALSE_BELIEF = "teammate_has_false_belief"
    TEAMMATE_HAS_TRUE_BELIEF = "teammate_has_true_belief"
    TEAMMATE_HAS_NO_BELIEF = "teammate_has_no_belief"
    PLAYER_HAS_UNCERTAINTY = "player_has_uncertainty"
    PLAYER_HAS_NO_BELIEF = "player_has_no_belief"
    OPPONENT_HAS_FALSE_BELIEF = "opponent_has_false_belief"
    OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY = "opponent_has_true_belief_with_uncertainty"
    HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "honest_opponent_has_true_belief_with_uncertainty"
    DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY = "dishonest_opponent_has_true_belief_with_uncertainty"


class AskConstraintType(Enum):
    TEAMMATE_LACKS_KNOWLEDGE = "teammate_lacks_knowledge"
    TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE = "teammate_and_neutral_lack_knowledge"
    TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE = "teammate_neutral_and_honest_opponent_lack_knowledge"
    NO_CONSTRAINT = "no_constraint"


@dataclass
class Character:
    name: str
    team: Team
    char_type: CharacterType
    knowledge: Dict[str, Optional[str]] = field(default_factory=dict)
    beliefs: Dict[str, Optional[str]] = field(default_factory=dict)
    
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


@dataclass
class Event:
    """Represents an event in a scenario."""
    event_type: str  # 'put', 'move', 'leave'
    character: str
    container: Optional[str] = None
    item: Optional[str] = None
    from_container: Optional[str] = None
    to_container: Optional[str] = None


@dataclass
class Scenario:
    """Represents one scenario in the game."""
    round_num: int
    whose_turn: str
    who_answers: str
    question_container: str
    events: List[Event]
    present_initially: List[str]
    epistemic_type: Optional[EpistemicType] = None
    ask_constraint: Optional[AskConstraintType] = None
    
    def get_description_for(self, character_name: str, characters: Dict[str, Character]) -> str:
        """Generate scenario description from a character's perspective."""
        char_map = {c: c for c in characters.keys()}
        if character_name in characters:
            char_map[character_name] = "You"
        
        lines = []
        present = set(self.present_initially)
        
        # Initial state
        def format_name_list(names: List[str]) -> str:
            """Return 'X', 'X and Y', or 'X, Y, and Z' with an Oxford comma."""
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            if len(names) == 2:
                return f"{names[0]} and {names[1]}"
            return ", ".join(names[:-1]) + f", and {names[-1]}"
        names = [char_map.get(c, c) for c in sorted(present)]
        for i, name in enumerate(names):
            if i>0 and name == "You": names[i] = "you"
        char_list = format_name_list(names)
        verb = "are" if (len(names) > 1 or names[0].lower() == "you") else "is"
        lines.append(f"{char_list} {verb} in a room. Inside the room are an empty bag and an empty box.")

        # Narrate only while the perspective character is present
        perspective_present = character_name in present

        for event in self.events:
            if not perspective_present:
                break

            actor = char_map.get(event.character, event.character)
            you_form = (actor == "You")

            if event.event_type == 'put':
                verb_put = "put" if you_form else "puts"
                lines.append(f"{actor} {verb_put} a {event.item} in the {event.container}.")

            elif event.event_type == 'move':
                verb_move = "move" if you_form else "moves"
                lines.append(f"{actor} {verb_move} the {event.item} to the {event.to_container}.")

            elif event.event_type == 'leave':
                verb_leave = "leave" if you_form else "leaves"
                lines.append(f"{actor} {verb_leave} the room.")
                present.discard(event.character)
                if event.character == character_name:
                    # Stop narrating once the perspective person leaves
                    perspective_present = False
                    break

        return " ".join(lines)
    
    def to_dict(self) -> dict:
        """Convert scenario to dictionary for JSON serialization."""
        return {
            'round_num': self.round_num,
            'whose_turn': self.whose_turn,
            'who_answers': self.who_answers,
            'question_container': self.question_container,
            'events': [asdict(e) for e in self.events],
            'present_initially': self.present_initially,
            'epistemic_type': self.epistemic_type.value if self.epistemic_type else None,
            'ask_constraint': self.ask_constraint.value if self.ask_constraint else None
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Scenario':
        """Create scenario from dictionary."""
        return Scenario(
            round_num=data['round_num'],
            whose_turn=data['whose_turn'],
            who_answers=data['who_answers'],
            question_container=data['question_container'],
            events=[Event(**e) for e in data['events']],
            present_initially=data['present_initially'],
            epistemic_type=EpistemicType(data['epistemic_type']) if data.get('epistemic_type') else None,
            ask_constraint=AskConstraintType(data['ask_constraint']) if data.get('ask_constraint') else None
        )


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
    epistemic_type: Optional[str] = None
    ask_constraint: Optional[str] = None


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
                'B': Character('B', Team.BLUE, CharacterType.HONEST_TEAMMATE),
                'C': Character('C', Team.RED, CharacterType.HONEST_OPPONENT),
                'D': Character('D', Team.RED, CharacterType.DISHONEST_OPPONENT),
                'N': Character('N', Team.NEUTRAL, CharacterType.NEUTRAL),
            }
    
    def get_teammate(self, actor_name: str) -> Optional[str]:
        """Get the teammate of the actor."""
        actor = self.characters[actor_name]
        for name, char in self.characters.items():
            if name != actor_name and char.team == actor.team:
                return name
        return None
    
    def is_honest_to(self, speaker_name: str, listener_name: str) -> bool:
        """Check if speaker is honest to listener."""
        speaker = self.characters[speaker_name]
        listener = self.characters[listener_name]
        
        if speaker.char_type == CharacterType.NEUTRAL:
            return True
        if speaker.char_type == CharacterType.HONEST_OPPONENT:
            return True
        return speaker.team == listener.team
    
    def trusts(self, listener_name: str, speaker_name: str) -> bool:
        listener = self.characters[listener_name]
        speaker = self.characters[speaker_name]

        # Teammates always trust each other (including the live player)
        if listener.team != Team.NEUTRAL and listener.team == speaker.team:
            return True

        # Neutral trusts everyone
        if listener.char_type == CharacterType.NEUTRAL:
            return True

        # (Keep any remaining special cases you want)
        if listener.char_type == CharacterType.HONEST_OPPONENT:
            return True
        if listener.char_type == CharacterType.HONEST_TEAMMATE:
            return True
        if listener.char_type == CharacterType.DISHONEST_OPPONENT:
            return speaker.team == listener.team
        if listener.char_type == CharacterType.DISHONEST_TEAMMATE:
            return speaker.team == listener.team

        return False
        
    def reset_scenario_knowledge(self):
        """Reset all character knowledge for new scenario."""
        for char in self.characters.values():
            char.knowledge = {}
            char.beliefs = {}
    
    def get_present_at_end(self, scenario: Scenario) -> Set[str]:
        """Get who's present at the end of the scenario."""
        present = set(scenario.present_initially)
        for event in scenario.events:
            if event.event_type == 'leave':
                present.discard(event.character)
        return present
    
    def process_scenario_events(self, scenario: Scenario):
        """Process scenario events to determine what each character knows."""
        self.reset_scenario_knowledge()
        
        present = set(scenario.present_initially)
        container_contents = {'bag': None, 'box': None}
        
        for event in scenario.events:
            if event.event_type == 'put':
                container_contents[event.container] = event.item
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.container, event.item)
                    
            elif event.event_type == 'move':
                container_contents[event.to_container] = event.item
                container_contents[event.from_container] = None
                for char_name in present:
                    self.characters[char_name].update_knowledge(event.to_container, event.item)
                    self.characters[char_name].update_knowledge(event.from_container, None)
                        
            elif event.event_type == 'leave':
                present.discard(event.character)
        
        return container_contents
    
    def execute_npc_action(self, actor_name: str, scenario: Scenario, true_contents: Dict[str, str]) -> Action:
        """Execute NPC action based on game rules - completely determined."""
        container = scenario.question_container
        truth = true_contents[container]
        target = scenario.who_answers
        
        # Tell teammate truth scenarios
        if scenario.epistemic_type in [EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
                                         EpistemicType.TEAMMATE_HAS_NO_BELIEF]:
            return Action(ActionType.TELL, target, container, truth)
        
        # Pass scenarios
        if scenario.epistemic_type in [EpistemicType.OPPONENT_HAS_FALSE_BELIEF]:
            return Action(ActionType.PASS)
        
        if scenario.epistemic_type in [EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY,
                                         EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY]:
            return Action(ActionType.PASS)
        
        if scenario.epistemic_type in [EpistemicType.TEAMMATE_HAS_TRUE_BELIEF]:
            return Action(ActionType.PASS)
        
        # Tell opponent scenario - tell lie only if actor is dishonest to target
        if scenario.epistemic_type == EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
            if self.characters[actor_name].char_type in [CharacterType.HONEST_OPPONENT, CharacterType.HONEST_TEAMMATE]:
                return Action(ActionType.PASS)
            else:
                possible_lies = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
                lie = next((item for item in possible_lies if item != truth), 'nothing')
                return Action(ActionType.TELL, target, container, lie)
        
        # Ask scenarios: ask based on who's present
        if scenario.epistemic_type in [EpistemicType.PLAYER_HAS_UNCERTAINTY,
                                         EpistemicType.PLAYER_HAS_NO_BELIEF]:
            present = self.get_present_at_end(scenario)
            present.discard(actor_name)
            
            # Preference: teammate -> neutral -> opponent (if trusts opponents)
            teammate = self.get_teammate(actor_name)
            if teammate and teammate in present:
                return Action(ActionType.ASK, teammate, container)
            
            if 'N' in present:
                return Action(ActionType.ASK, 'N', container)
            
            # Ask opponent only if actor trusts opponents
            actor = self.characters[actor_name]
            actor_trusts_opponents = actor.char_type in [CharacterType.HONEST_TEAMMATE, 
                                                         CharacterType.HONEST_OPPONENT]
            if actor_trusts_opponents:
                # Choose a random eligible opponent from those present
                eligible = [name for name in present
                            if name != actor_name and self.characters[name].char_type in [
                                CharacterType.HONEST_TEAMMATE,
                                CharacterType.DISHONEST_TEAMMATE,
                                CharacterType.HONEST_OPPONENT,
                                CharacterType.DISHONEST_OPPONENT
                            ]]
                if eligible:
                    return Action(ActionType.ASK, random.choice(eligible), container)
            
            return Action(ActionType.PASS)
        
        return Action(ActionType.PASS)
    
    def is_action_optimal(self, action_str: str, scenario: Scenario, true_contents: Dict[str, str]) -> bool:
        """Check if live player's action is optimal."""
        action = self.parse_action(action_str)
        if not action:
            return False
        
        # For Tell teammate truth scenarios
        if scenario.epistemic_type in [EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
                                         EpistemicType.TEAMMATE_HAS_NO_BELIEF]:
            truth = true_contents[scenario.question_container]
            return (action.action_type == ActionType.TELL and
                   action.target_char == scenario.who_answers and
                   action.container == scenario.question_container and
                   action.contents == truth)
        
        # For Tell opponent with uncertainty
        if scenario.epistemic_type == EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
            truth = true_contents[scenario.question_container]
            if action.action_type != ActionType.TELL:
                return False
            if action.target_char != scenario.who_answers or action.container != scenario.question_container:
                return False
            return action.contents != truth
        
        # For Ask scenarios
        if scenario.epistemic_type in [EpistemicType.PLAYER_HAS_UNCERTAINTY,
                                         EpistemicType.PLAYER_HAS_NO_BELIEF]:
            if scenario.ask_constraint == AskConstraintType.TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE:
                return action.action_type == ActionType.PASS
            
            if action.action_type == ActionType.TELL or action.container != scenario.question_container:
                return False
            
            present = self.get_present_at_end(scenario)
            present.discard(scenario.whose_turn)
            
            if action.target_char not in present:
                return False
            
            if scenario.ask_constraint == AskConstraintType.NO_CONSTRAINT:
                return self.characters[action.target_char].char_type in [CharacterType.HONEST_TEAMMATE, CharacterType.DISHONEST_TEAMMATE]
            elif scenario.ask_constraint == AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE:
                return self.characters[action.target_char].char_type in [CharacterType.NEUTRAL]
            elif scenario.ask_constraint == AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE:
                return self.characters[action.target_char].char_type in [CharacterType.HONEST_OPPONENT]
            else: 
                return False
        
        # For Pass scenarios
        if scenario.epistemic_type in [EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,
                                         EpistemicType.OPPONENT_HAS_FALSE_BELIEF,
                                         EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY,
                                         EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY]:
            return action.action_type == ActionType.PASS
        
        return False
    
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
        
        ask_match = re.match(r'Ask\(([A-DN]),\s*(bag|box)\)', action_str, re.IGNORECASE)
        if ask_match:
            return Action(ActionType.ASK, ask_match.group(1).upper(), ask_match.group(2).lower())
        
        tell_match = re.match(r'Tell\(([A-DN]),\s*(bag|box),\s*(\w+)\)', action_str, re.IGNORECASE)
        if tell_match:
            return Action(ActionType.TELL, tell_match.group(1).upper(), 
                         tell_match.group(2).lower(), tell_match.group(3).lower())
        
        return None
    
    def execute_action(self, actor_name: str, action: Action, true_contents: Dict[str, str]) -> Tuple[float, str]:
        """Execute an action and return the score change and description."""
        score_delta = 0.0
        desc = ""
        
        if action.action_type == ActionType.PASS:
            desc = "Pass"
            
        elif action.action_type == ActionType.ASK:
            score_delta -= 0.5
            target = self.characters[action.target_char]
            container = action.container

            # What the target says:
            # - Honest: report what they believe (what they observed).
            # - Dishonest: intentionally say something different from their belief.
            if target.char_type in [CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT]:
                believed = target.get_belief(container)
                options = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
                # Pick an item different from what they believe; if they believe nothing, pick any item.
                wrong = next((x for x in options if x != (believed or 'nothing')), options[0])
                answer = wrong
            else:
                answer = target.get_belief(container)

            # Asking always updates the asker’s belief with whatever was answered.
            self.characters[actor_name].receive_info(container, answer, target, True)

            desc = f"Ask({action.target_char}, {container})"
            
        elif action.action_type == ActionType.TELL:
            score_delta -= 0.5
            target_name = action.target_char
            
            if self.trusts(target_name, actor_name):
                self.characters[target_name].receive_info(action.container, action.contents,
                                                         self.characters[actor_name], True)
            
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
        if self.scores[Team.BLUE] >= WINNING_SCORE:
            self.game_over = True
            self.winner = Team.BLUE
        elif self.scores[Team.RED] >= WINNING_SCORE:
            self.game_over = True
            self.winner = Team.RED
        elif self.current_scenario_idx >= len(self.scenarios):
            self.game_over = True
            if self.scores[Team.BLUE] > self.scores[Team.RED]:
                self.winner = Team.BLUE
            elif self.scores[Team.RED] > self.scores[Team.BLUE]:
                self.winner = Team.RED
            else:
                self.winner = None
    
    def advance_turn(self):
        """Move to next turn."""
        self.current_scenario_idx += 1
        self.current_turn_idx += 1
        if self.current_turn_idx >= len(self.turn_order):
            self.current_turn_idx = 0


def load_scenarios(filename: str) -> List[Scenario]:
    """Load scenarios from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return [Scenario.from_dict(s) for s in data]


def save_scenarios(scenarios: List[Scenario], filename: str):
    """Save scenarios to JSON file."""
    with open(filename, 'w') as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=2)


def save_game_results(turn_records: List[TurnRecord], filename: str):
    """Save game results to JSON file."""
    with open(filename, 'w') as f:
        json.dump([asdict(r) for r in turn_records], f, indent=2)


def create_game(filename=None) -> GameState:
    if filename is None:
        scenarios = get_default_scenarios()
        save_scenarios(scenarios, 'scenarios.json')
        print("Created scenarios.json with default scenarios")
    else:
        try:
            scenarios = load_scenarios(filename)
        except FileNotFoundError:
            # If file doesn't exist, create default scenarios and save them
            scenarios = get_default_scenarios()
            save_scenarios(scenarios, 'scenarios.json')
            print("Scenario file not found. Created scenarios.json with default scenarios")
    
    turn_order = ['A', 'D', 'B', 'C']
    
    game = GameState(scenarios, turn_order)
    game.initialize_characters('A')
    
    return game


def get_default_scenarios() -> List[Scenario]:
    """Get the default 12 scenarios."""
    return [
        # Round 1, Live Player turn
        Scenario(
            epistemic_type=EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
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
            epistemic_type=EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
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
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
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
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
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
            epistemic_type=EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
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
            epistemic_type=EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
            round_num=2,
            whose_turn='D',
            who_answers='C',
            question_container='box',
            present_initially=['D', 'A', 'B', 'C', 'N'],
            events=[
                Event('put', 'N', 'box', 'ball'),
                Event('leave', 'C'),
                Event('move', 'N', to_container='bag', item='ball', from_container='box'),
                Event('put', 'N', 'box', 'banana'),
                Event('leave', 'B'),
            ]
        ),
        # Round 2, B turn
        Scenario(
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE,
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
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE,
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
            epistemic_type=EpistemicType.TEAMMATE_HAS_NO_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
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
            epistemic_type=EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
            ask_constraint=AskConstraintType.NO_CONSTRAINT,
            round_num=3,
            whose_turn='D',
            who_answers='C',
            question_container='bag',
            present_initially=['D', 'A', 'B', 'C', 'N'],
            events=[
                Event('put', 'N', 'box', 'ball'),
                Event('leave', 'C'),
                Event('move', 'N', to_container='bag', item='ball', from_container='box'),
                Event('put', 'N', 'box', 'banana'),
                Event('leave', 'B'),
            ]
        ),
        # Round 3, B turn
        Scenario(
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE,
            round_num=3,
            whose_turn='B',
            who_answers='B',
            question_container='bag',
            present_initially=['B', 'A', 'C', 'D', 'N'],
            events=[
                Event('put', 'B', 'bag', 'ball'),
                Event('leave', 'B'),
                Event('leave', 'A'),
                Event('leave', 'C'),
                Event('leave', 'N'),
            ]
        ),
        # Round 3, C turn
        Scenario(
            epistemic_type=EpistemicType.PLAYER_HAS_UNCERTAINTY,
            ask_constraint=AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE,
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


def play_game_cli(scenario_file: str = None, human_player: bool = True):
    """Play the game in CLI mode."""
    game = create_game(scenario_file)
    global WINNING_SCORE
    WINNING_SCORE = max(WINNING_SCORE, int(len(game.scenarios)/len(game.turn_order)) + 4)
    
    print("=" * 70)
    print(GAME_SETUP.format(WINNING_SCORE=WINNING_SCORE))
    print("=" * 70)
    
    turn_count = 0

    while not game.game_over and game.get_current_scenario():
        scenario = game.get_current_scenario()
        turn_char = game.get_current_turn_character()
        
        if not turn_char:
            break
        
        true_contents = game.process_scenario_events(scenario)
        
        print("\n***********************************")
        print(f"Score: Blue={game.scores[Team.BLUE]}, Red={game.scores[Team.RED]}")
        if turn_count == 0:
            print("Here's the first scenario:")
        else:
            print(f"Here's scenario {turn_count + 1}:")
        
        turn_name = "Your" if turn_char == 'A' else f"{turn_char}'s"
        print(f"{turn_name} turn:")
        
        scenario_desc = scenario.get_description_for(turn_char, game.characters)
        answerer = "you" if scenario.who_answers == turn_char else scenario.who_answers
        question_desc = f"I am going to ask {answerer} what is in the {scenario.question_container}."
        
        print("-----------------------------------------------")
        print(scenario_desc)
        print("----------------------------------------------")
        print(question_desc)
        print("Respond ONLY with your action, and no other text.")
        
        action = None
        action_str = ""
        if turn_char == 'A':
            while not action:
                action_str = input(f"Your action (Ask(Player, Container), Tell(Player, Container, Contents), or Pass): ")
                action = game.parse_action(action_str)
                if not action:
                    print("Invalid action format. Try again.")
        else:
            action = game.execute_npc_action(turn_char, scenario, true_contents)
            if action.action_type == ActionType.ASK:
                action_str = f"Ask({action.target_char}, {action.container})"
            elif action.action_type == ActionType.TELL:
                action_str = f"Tell({action.target_char}, {action.container}, {action.contents})"
            else:
                action_str = "Pass"
        
        score_delta, action_desc = game.execute_action(turn_char, action, true_contents)
                
        print(f"\nAction: {action_str}")
        
        # Answer phase
        answer_given, is_correct, answer_score = game.resolve_answer_phase(scenario, true_contents)
        
        # Display answer
        print(f"{scenario.who_answers} answers: {answer_given}")
        if is_correct:
            print(f"Correct! The {scenario.question_container} contains {answer_given}.")
        else:
            print(f"Incorrect. The {scenario.question_container} actually contains {true_contents[scenario.question_container]}.")
        
        # Compute per-team deltas
        blue_delta = 0.0
        red_delta = 0.0

        # Action cost applies to acting player's team
        if turn_char in ['A', 'B']:
            blue_delta += score_delta
        else:
            red_delta += score_delta

        # Answer points apply only to the answerer's team (if correct)
        if is_correct:
            if scenario.who_answers in ['A', 'B']:
                blue_delta += answer_score
            else:
                red_delta += answer_score

        # Apply deltas to the scoreboard
        game.scores[Team.BLUE] += blue_delta
        game.scores[Team.RED] += red_delta

        # Outcome message shows exact deltas for both teams
        def fmt_delta(x: float) -> str:
            sign = '+' if x >= 0 else '-'
            return f"{sign}{abs(x)}"
        print(f"\nOutcome: Blue {fmt_delta(blue_delta)}, Red {fmt_delta(red_delta)}")

        # Check if action was optimal (only for live player)
        was_optimal = False
        expected_action_str = ""
        if turn_char == 'A':
            was_optimal = game.is_action_optimal(action_str, scenario, true_contents)
            expected_action_obj = game.execute_npc_action(turn_char, scenario, true_contents)
            if expected_action_obj.action_type == ActionType.PASS:
                expected_action_str = "Pass"
            elif expected_action_obj.action_type == ActionType.ASK:
                expected_action_str = f"Ask({expected_action_obj.target_char}, {expected_action_obj.container})"
            elif expected_action_obj.action_type == ActionType.TELL:
                expected_action_str = f"Tell({expected_action_obj.target_char}, {expected_action_obj.container}, {expected_action_obj.contents})"
        else:
            was_optimal = True
            expected_action_str = action_str
        
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
            optimal_action=expected_action_str,
            was_optimal=was_optimal,
            blue_score_after=game.scores[Team.BLUE],
            red_score_after=game.scores[Team.RED],
            epistemic_type=scenario.epistemic_type.value if scenario.epistemic_type else None,
            ask_constraint=scenario.ask_constraint.value if scenario.ask_constraint else None
        )
        game.turn_records.append(turn_record)
        
        # Wait for user to press space (if human player)
        if human_player:
            input("\n[Press Enter to continue]")
        
        game.check_game_over()
        game.advance_turn()
        turn_count += 1

    # Game over
    print("\n" + "=" * 70)
    print("GAME OVER")
    print(f"Final Score: Blue {game.scores[Team.BLUE]} - Red {game.scores[Team.RED]}")
    game.winner = "Blue" if game.scores[Team.BLUE] > game.scores[Team.RED] else "Red" if game.scores[Team.RED] > game.scores[Team.BLUE] else None
    if game.winner:
        print(f"Winner: {game.winner.value} team")
    elif game.winner is None:
        print("It's a tie!")
    print("=" * 70)
    """
    # Show turn records
    print("\n" + "=" * 70)
    print("TURN RECORD")
    print("=" * 70)
    for record in game.turn_records:
        print(f"\nRound {record.round_num} - {record.character}'s turn")
        print(f"Ontological Type: {record.epistemic_type}")
        print(f"Ask Constraint: {record.ask_constraint}")
        print(f"Action: {record.action}")
        if record.character == 'A':
            print(f"Expected: {record.optimal_action}")
            print(f"Was Expected: {'YES' if record.was_optimal else 'NO'}")
        print(f"Answer Given: {record.answer_given}")
        print(f"Answer Correct: {'YES' if record.answer_correct else 'NO'}")
        print(f"Score After: Blue {record.blue_score_after} - Red {record.red_score_after}")
    """ 
    # Save results
    save_game_results(game.turn_records, 'game_results.json')
    print("\nGame results saved to game_results.json")
    
    return game


if __name__ == "__main__":
    play_game_cli(scenario_file = 'scenarios_generated2.json', human_player=True)