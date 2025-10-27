import random
from tom_helpers import (
    Scenario, Event, EpistemicType, AskConstraintType, CharacterType,
    save_scenarios
)
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

SpecTuple = Tuple['EpistemicType', Optional['AskConstraintType'], 'CharacterType']

ITEMS_GEN = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
CONTAINERS_GEN = ['bag', 'box']

def _map_chartypes_to_names(chartypes: List[CharacterType]) -> List[str]:
    chars = []
    for ct in chartypes:
        if ct == CharacterType.LIVE_PLAYER: chars.append('A')
        elif ct in [CharacterType.HONEST_TEAMMATE, CharacterType.DISHONEST_TEAMMATE]: chars.append('B')
        elif ct == CharacterType.NEUTRAL: chars.append('N')
        elif ct == CharacterType.HONEST_OPPONENT and 'C' not in chars: chars.append('C')
        elif ct == CharacterType.HONEST_OPPONENT and 'C' in chars: chars.append('D')
        elif ct == CharacterType.DISHONEST_OPPONENT and CharacterType.HONEST_OPPONENT not in chartypes and 'C' not in chars: chars.append('C')
        elif ct == CharacterType.DISHONEST_OPPONENT: chars.append('D')
    return chars

def _teammate_of(name: str) -> str:
    return {'A': 'B', 'B': 'A', 'C': 'D', 'D': 'C'}[name]

def _opponent_of(name: str, rng: random.Random) -> str:
    #Get a random opponent of the given character.
    if name in ['A', 'B']:
        return rng.choice(['C', 'D'])  # Blue team → pick a red opponent
    else:
        return rng.choice(['A', 'B'])  # Red team → pick a blue opponent

def _other_container(c: str) -> str:
    return CONTAINERS_GEN[1] if c == CONTAINERS_GEN[0] else CONTAINERS_GEN[0]

def _pick_other_item(rng: random.Random, exclude: str) -> str:
    return rng.choice([x for x in ITEMS_GEN if x != exclude])

@dataclass
class _Builder:
    """Presence-safe builder that ensures valid moves and no actions after leaving."""
    rng: random.Random
    qc: str                # question container
    available: Set[str]    # who is allowed to be in the room initially

    def __post_init__(self):
        self.present: Set[str] = set(self.available)
        self.events: List[Event] = []
        self.contents = {c: None for c in CONTAINERS_GEN} 
        self.used: Set[str] = set()  # anyone who acts or leaves

    def rand_actor(self, exclude: Optional[Set[str]] = None) -> str:
        pool = [p for p in self.present if not exclude or p not in exclude]
        return self.rng.choice(pool)

    def leave(self, who: str):
        # Schedule a leave only if they’re currently present
        if who in self.present:
            self.events.append(Event('leave', who))
            self.present.discard(who)
            self.used.add(who)

    def _move_out_if_needed(self, container: str, who: str):
        existing = self.contents[container]
        if existing is None:
            return
        to_cont = _other_container(container)
        # In this generator we keep 'to_cont' empty until needed
        if self.contents[to_cont] is not None:
            # Safety: shouldn't happen in our single-flip patterns
            return
        self.events.append(Event('move', who, from_container=container, to_container=to_cont, item=existing))
        self.used.add(who)
        self.contents[container] = None
        self.contents[to_cont] = existing

    def put(self, who: str, container: str, item: str, exclude: Optional[Set[str]] = None):
        # Ensure we never narrate simultaneous items in a container
        if self.contents[container] is not None and self.contents[container] != item:
            self._move_out_if_needed(container, self.rand_actor(exclude))
        self.contents[container] = item
        self.events.append(Event('put', who, container=container, item=item))
        self.used.add(who)

    def put_random(self, item: str, exclude: Optional[Set[str]] = None):
        self.put(self.rand_actor(exclude), self.qc, item, exclude)


def _identify_who_answers(actor: str, ent: 'EpistemicType', rng: random.Random) -> str:
    # Opponent choice is uniform between C and D when generic
    if ent in (EpistemicType.PLAYER_HAS_CERTAINTY, EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF):
        return actor
    if ent in (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
               EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,
               EpistemicType.TEAMMATE_HAS_UNKNOWN_BELIEF,
               EpistemicType.TEAMMATE_HAS_NO_BELIEF):
        return _teammate_of(actor)
    if ent == EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        return 'C'
    if ent == EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        return 'D'
    if ent in (EpistemicType.OPPONENT_HAS_FALSE_BELIEF,
               EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY):
        return rng.choice(['C', 'D'])
    return rng.choice(['C', 'D'])


def _plan_availability_for_player_ask(actor: str, ask: AskConstraintType, chars: List[str]) -> Tuple[Set[str], Optional[str]]:
    """
    Returns:
      - available: who is allowed to be present initially (others never present).
      - intended_knower: who we intend to be present and witness final truth (None if not constrained).
    Note: This matches is_action_optimal’s absolute-type checks:
      - NO_CONSTRAINT: teammate types => in this cast B is the only HONEST_TEAMMATE
      - TEAMMATE_LACKS_KNOWLEDGE: ask N
      - TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE: ask HONEST_OPPONENT (C)
      - TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE: PASS is optimal (we can let D know or no one)
    """

    exclude: Set[str] = set()

    # Remove higher-priority options by never including them
    if ask == AskConstraintType.NO_CONSTRAINT:
        # is_action_optimal expects asking a teammate-type; in your cast that's B
        intended_knower = _teammate_of(actor)
        # No removals needed; we want B eligible and knowledgeable

    elif ask == AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE:
        # Remove actor’s teammate entirely so they can’t be asked and won’t know
        exclude.add(_teammate_of(actor))
        intended_knower = 'N' if 'N' in chars else 'C' if 'C' in chars else 'D'  # absolute NEUTRAL if available, else HONEST_OPPONENT, else DISHONEST_OPPONENT

    elif ask == AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE:
        exclude.add(_teammate_of(actor))
        exclude.add('N')
        intended_knower = 'C' if 'C' in chars else 'D'  # absolute HONEST_OPPONENT if available, else DISHONEST_OPPONENT

    elif ask == AskConstraintType.TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE:
        exclude.add(_teammate_of(actor))
        exclude.add('N')
        exclude.add('C')
        intended_knower = 'D'  # absolute DISHONEST_OPPONENT

    return exclude, intended_knower

def _build_answerer_belief(sb: _Builder,
                           ent: 'EpistemicType',
                           answerer: str,
                           final_item: str,
                           rng: random.Random):
    """
    Construct events to realize the desired belief state for the answerer.
    Only change the asked container; keep other container as a parking spot when flipping.
    """
    if ent in (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
               EpistemicType.OPPONENT_HAS_FALSE_BELIEF):
        # Witness initial put X, then leave; flip to final after they leave.
        old_item = _pick_other_item(rng, final_item)
        sb.put_random(old_item)
        sb.leave(answerer)
        # For TEAMMATE scenarios, exclude the player from acting after teammate leaves
        exclude_set = {_teammate_of(answerer)} if ent == EpistemicType.TEAMMATE_HAS_FALSE_BELIEF else None
        sb.put_random(final_item, exclude=exclude_set)

    elif ent in (EpistemicType.TEAMMATE_HAS_NO_BELIEF, EpistemicType.PLAYER_HAS_NO_BELIEF):
        # Answerer absent before any changes: if present, make them leave immediately, else never present.
        sb.leave(answerer)
        # For TEAMMATE scenarios, exclude the player from acting
        exclude_set = {_teammate_of(answerer)} if ent == EpistemicType.TEAMMATE_HAS_NO_BELIEF else None
        sb.put_random(final_item, exclude=exclude_set)

    elif ent in (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,):
        # Answerer witnesses final truth; may or may not leave afterward; no further changes.
        sb.put_random(final_item)
        if rng.random() < 0.5:
            # Optional leave after final truth (no further changes)
            sb.leave(answerer)

    elif ent in [EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY, EpistemicType.PLAYER_HAS_CERTAINTY]:
        # Must not leave; witness final truth and remain present until end; no further changes.
        sb.put_random(final_item)
        # Do not schedule a leave for the answerer

    elif ent in [EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY, EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY]:
        sb.put_random(final_item)
        sb.leave(answerer)

    elif ent == EpistemicType.PLAYER_HAS_UNCERTAINTY:
        # Actor (who equals answerer) witnesses an initial put, then leaves; then flip the truth.
        old_item = _pick_other_item(rng, final_item)
        sb.put_random(old_item)   # actor present here
        sb.leave(answerer)        # actor leaves (now uncertain)
        # To make the belief “not guaranteed to be correct”, perform a flip after they leave.
        if rng.random() < 0.5:
            sb.put_random(final_item)

    elif ent == EpistemicType.TEAMMATE_HAS_UNKNOWN_BELIEF:
        old_item = _pick_other_item(rng, final_item)
        sb.put_random(old_item)   # actor and teammate present here
        sb.leave(answerer)         # teammate leaves (now uncertain)
        sb.leave(_teammate_of(answerer)) #actor leaves (now uncertain)
        # To make the belief “not guaranteed to be correct”, perform a flip after they leave.
        if rng.random() < 0.5:
            sb.put_random(final_item)


def _validate_invariants(s: 'Scenario') -> None:
    """
    Defensive checks to catch logical errors early:
      - No one acts after leaving.
      - Asked container never narrates two different items without an intervening move.
    """
    present = set(s.present_initially)
    contents = {'bag': None, 'box': None}
    for idx, e in enumerate(s.events):
        if e.event_type == 'leave':
            present.discard(e.character)
        elif e.event_type == 'put':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            if contents[e.container] is not None and contents[e.container] != e.item:
                # Should be immediately preceded by a move out of that container
                if not (idx > 0 and s.events[idx-1].event_type == 'move'
                        and s.events[idx-1].from_container == e.container):
                    raise ValueError(f"Event {idx}: put into non-empty {e.container} without prior move.")
            contents[e.container] = e.item
        elif e.event_type == 'move':
            if e.character not in present:
                raise ValueError(f"Event {idx}: {e.character} acted after leaving.")
            contents[e.to_container] = e.item
            contents[e.from_container] = None


def generate_scenarios_from_tuples(specs: List[SpecTuple], outfile: str, seed: Optional[int] = None, chartypes: List[CharacterType] = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT, CharacterType.NEUTRAL]) -> None:
    """
    Generate scenarios from (EpistemicType, AskConstraintType|None, CharacterType) tuples.

    Guarantees:
      - Uniform randomization for choices (who acts, which opponent answers generically, whether optional leave occurs).
      - Minimal initial presence: only characters required to act/leave or be present at end are included.
      - Presence-safe: no actions by characters after they leave.
      - Asked container changes never narrate two items without a move-out first.
      - Certainty semantics:
          * ...WITH_CERTAINTY: answerer remains present through end.
          * ...TRUE_BELIEF (no certainty): answerer may leave after final truth; no further asked-container changes after they leave.
      - PLAYER_HAS_UNCERTAINTY: actor forms a belief, leaves, and then we optionally flip the asked container after they leave (so belief is not guaranteed to be correct).
      - AskConstraint for PLAYER_*: we remove higher-priority targets by never including them and ensure the intended target is present to witness final truth (when applicable).
    """
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    chars = _map_chartypes_to_names(chartypes)
    acting_chars = [c for c in chars if c != 'N']

    for i, (ent, ask, actor_role) in enumerate(specs):
        actor = acting_chars[i % len(acting_chars)]
        qc = rng.choice(CONTAINERS_GEN)
        answerer = _identify_who_answers(actor, ent, rng)

        # Plan availability (who can be present initially)
        available: Set[str] = set(chars)

        intended_knower: Optional[str] = None
        present_initially: Set[str] = set(actor)

        # Enforce AskConstraint only for PLAYER_* ontologies
        if ent in (EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF):
            exclude, intended_knower = _plan_availability_for_player_ask(actor, ask, chars)
            available.difference_update(exclude)
            present_initially.add(intended_knower)

            # Ensure at least one opponent is available and present initially because actor cannot be the last person in the room
            opponent = _opponent_of(actor, rng)
            available.add(opponent)
            present_initially.add(opponent)  # Guarantee they're in the room!

        # Certainty semantics: for ...WITH_CERTAINTY, answerer must be present through end
        elif ent in [EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY, EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY, EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY]:
            present_initially.add(answerer)

        # Ensure at least one opponent is available and present initially because the opponent must be able to move things after the teammate leaves
        elif ent in (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, EpistemicType.TEAMMATE_HAS_NO_BELIEF):
            opponent = _opponent_of(actor, rng)
            available.add(opponent)
            present_initially.add(opponent)

        # Build events with a presence-safe builder
        sb = _Builder(rng, qc, available)
        final_item = rng.choice(ITEMS_GEN)

        # Build belief state for the answerer
        _build_answerer_belief(sb, ent, answerer, final_item, rng)

        # If this is a PLAYER_* Ask scenario, ensure the intended knower witnesses the final truth
        if ent in [EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF]:
            if sb.contents[sb.qc] != final_item:
                ##print(f"Ensuring {intended_knower} witnesses final truth {final_item}, ent={ent}, ask={ask}, actor={actor}, answerer={answerer}, final_item={final_item}, contents of target container={sb.contents[sb.qc]}")
                sb.put_random(final_item)

        # Minimal initial presence
        present_initially = present_initially.union(sb.used)
        ### present_initially must include at least one person who will remain in the room at the end to witness the final truth
        present_initially = sorted(list(present_initially))

        scenario = Scenario(
            round_num=(i // len(acting_chars)) + 1,
            whose_turn=actor,
            who_answers=answerer,
            question_container=qc,
            events=sb.events,
            present_initially=present_initially,
            epistemic_type=ent,
            ask_constraint=ask
        )

        # Validate invariants
        _validate_invariants(scenario)

        scenarios.append(scenario)

    save_scenarios(scenarios, outfile, chars, chartypes)


if __name__ == "__main__":
    # Auto-generate scenarios from tuple specs if not found
    specs: List[SpecTuple] = [
        # Round 1 (A, D, B, C)
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 2
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 2
        (EpistemicType.TEAMMATE_HAS_UNKNOWN_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 3
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_OPPONENT),

        # Round 4
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 5
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 6
        (EpistemicType.PLAYER_HAS_UNCERTAINTY, AskConstraintType.TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 7
        (EpistemicType.OPPONENT_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 8
        (EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 9
        (EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

        # Round 10
        (EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY, AskConstraintType.NO_CONSTRAINT, CharacterType.LIVE_PLAYER),
        (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.DISHONEST_OPPONENT),
        (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, AskConstraintType.NO_CONSTRAINT, CharacterType.HONEST_TEAMMATE),
        (EpistemicType.TEAMMATE_HAS_NO_BELIEF, AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE, CharacterType.HONEST_OPPONENT),

    ]
    outfile = 'scenarios_generated3.json'
    chartypes = [CharacterType.LIVE_PLAYER, CharacterType.HONEST_OPPONENT, CharacterType.DISHONEST_TEAMMATE, CharacterType.DISHONEST_OPPONENT, CharacterType.NEUTRAL]
    generate_scenarios_from_tuples(specs, outfile, seed=None, chartypes=chartypes)
    print(f"Created {outfile} with auto-generated scenarios")
