import random
from tom_test import (
    Scenario, Event, EpistemicType, AskConstraintType, CharacterType,
    save_scenarios
)
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

SpecTuple = Tuple['EpistemicType', Optional['AskConstraintType'], 'CharacterType']

ITEMS_GEN = ['apple', 'ball', 'banana', 'brick', 'stapler', 'orange']
CONTAINERS_GEN = ['bag', 'box']


def _actor_name_from_role(role: 'CharacterType') -> str:
    # Fixed mapping consistent with your initialize_characters
    mapping = {
        CharacterType.LIVE_PLAYER: 'A',
        CharacterType.HONEST_TEAMMATE: 'B',
        CharacterType.HONEST_OPPONENT: 'C',
        CharacterType.DISHONEST_OPPONENT: 'D',
    }
    if role not in mapping:
        raise ValueError(f"Unsupported actor role for generation: {role}")
    return mapping[role]


def _teammate_of(name: str) -> str:
    return {'A': 'B', 'B': 'A', 'C': 'D', 'D': 'C'}[name]


def _other_container(c: str) -> str:
    return 'box' if c == 'bag' else 'bag'


def _pick_other_item(rng: random.Random, exclude: str) -> str:
    return rng.choice([x for x in ITEMS_GEN if x != exclude])


def _uniform_choice(rng: random.Random, seq: List[str]) -> str:
    return rng.choice(seq)


@dataclass
class _Builder:
    """Presence-safe builder that ensures valid moves and no actions after leaving."""
    rng: random.Random
    qc: str                # question container
    available: Set[str]    # who is allowed to be in the room initially

    def __post_init__(self):
        # Start with only 'available' present; others never enter or act
        self.present_initially: Set[str] = set(self.available)
        self.present: Set[str] = set(self.available)
        self.events: List[Event] = []
        self.contents = {'bag': None, 'box': None}
        self.used: Set[str] = set()  # anyone who acts or leaves

    def rand_actor(self, exclude: Optional[Set[str]] = None) -> str:
        pool = [p for p in self.present if not exclude or p not in exclude]
        return self.rng.choice(pool) if pool else 'N'

    def leave(self, who: str):
        # Schedule a leave only if they’re currently present
        if who in self.present:
            self.events.append(Event('leave', who))
            self.present.discard(who)
            self.used.add(who)

    def _move_out_if_needed(self, container: str):
        existing = self.contents[container]
        if existing is None:
            return
        to_cont = _other_container(container)
        # In this generator we keep 'to_cont' empty until needed
        if self.contents[to_cont] is not None:
            # Safety: shouldn't happen in our single-flip patterns
            return
        who = self.rand_actor()
        self.events.append(Event('move', who, from_container=container, to_container=to_cont, item=existing))
        self.used.add(who)
        self.contents[container] = None
        self.contents[to_cont] = existing

    def put(self, who: str, container: str, item: str):
        # Ensure no one acts after leaving
        if who not in self.present:
            who = self.rand_actor()
        # Ensure we never narrate simultaneous items in a container
        if self.contents[container] is not None and self.contents[container] != item:
            self._move_out_if_needed(container)
        self.contents[container] = item
        self.events.append(Event('put', who, container=container, item=item))
        self.used.add(who)

    def put_random(self, item: str, exclude: Optional[Set[str]] = None):
        self.put(self.rand_actor(exclude), self.qc, item)


def _select_who_answers(actor: str, ent: 'EpistemicType', rng: random.Random) -> str:
    # Opponent choice is uniform between C and D when generic
    if ent in (EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF):
        return actor
    if ent in (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF,
               EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,
               EpistemicType.TEAMMATE_HAS_NO_BELIEF):
        return _teammate_of(actor)
    if ent == EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        return 'C'
    if ent == EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        return 'D'
    if ent in (EpistemicType.OPPONENT_HAS_FALSE_BELIEF,
               EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY):
        return _uniform_choice(rng, ['C', 'D'])
    return _uniform_choice(rng, ['C', 'D'])


def _plan_availability_for_player_ask(actor: str,
                                      ask: Optional['AskConstraintType']) -> Tuple[Set[str], Optional[str]]:
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
    everyone = {'A', 'B', 'C', 'D', 'N'}
    available = set(everyone)
    intended_knower: Optional[str] = None

    if not ask:
        return available, intended_knower

    # Remove higher-priority options by never including them
    if ask == AskConstraintType.NO_CONSTRAINT:
        # is_action_optimal expects asking a teammate-type; in your cast that's B
        intended_knower = 'B'
        # No removals needed; we want B eligible and knowledgeable

    elif ask == AskConstraintType.TEAMMATE_LACKS_KNOWLEDGE:
        # Remove actor’s teammate entirely so they can’t be asked and won’t know
        available.discard(_teammate_of(actor))
        intended_knower = 'N'

    elif ask == AskConstraintType.TEAMMATE_AND_NEUTRAL_LACK_KNOWLEDGE:
        available.discard(_teammate_of(actor))
        available.discard('N')
        intended_knower = 'C'  # absolute HONEST_OPPONENT

    elif ask == AskConstraintType.TEAMMATE_NEUTRAL_AND_HONEST_OPPONENT_LACK_KNOWLEDGE:
        available.discard(_teammate_of(actor))
        available.discard('N')
        available.discard('C')
        intended_knower = 'D'  # absolute DISHONEST_OPPONENT; optimal is Pass per your rule

    return available, intended_knower


def _ensure_intended_knower_witnesses(sb: _Builder, intended_knower: Optional[str], final_item: str):
    # If we want a specific person to know at the end, ensure they are present when final truth is (re)asserted.
    if not intended_knower or intended_knower not in sb.present:
        return
    # If the asked container already holds the final item, don’t emit a duplicate put
    if sb.contents[sb.qc] == final_item:
        return
    sb.put_random(final_item)

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
        # Ensure answerer is available to witness
        if answerer not in sb.present:
            # If absent (by design), switch to a no-belief pattern; but false belief requires presence.
            # Since ontology demands false belief, include them now (they must have been available).
            pass
        sb.put_random(old_item)
        sb.leave(answerer)
        sb.put_random(final_item)

    elif ent in (EpistemicType.TEAMMATE_HAS_NO_BELIEF,):
        # Answerer absent before any changes: if present, make them leave immediately, else never present.
        if answerer in sb.present:
            sb.leave(answerer)
        sb.put_random(final_item)

    elif ent == EpistemicType.PLAYER_HAS_NO_BELIEF:
        # Actor (who equals answerer) leaves before any change.
        sb.leave(answerer)
        sb.put_random(final_item)

    elif ent in (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,):
        # Answerer witnesses final truth; may or may not leave afterward; no further changes.
        sb.put_random(final_item)
        if rng.random() < 0.5:
            # Optional leave after final truth (no further changes)
            sb.leave(answerer)

    elif ent == EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY:
        # Must not leave; witness final truth and remain present until end; no further changes.
        sb.put_random(final_item)
        # Do not schedule a leave for the answerer

    elif ent == EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        # Witness final truth, then leave (uncertainty = not present through to end).
        sb.put_random(final_item)
        sb.leave(answerer)

    elif ent == EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY:
        # Same pattern: witness final truth, then leave.
        sb.put_random(final_item)
        sb.leave(answerer)

    elif ent == EpistemicType.PLAYER_HAS_UNCERTAINTY:
        # Actor (who equals answerer) witnesses an initial put, then leaves; then flip the truth.
        old_item = _pick_other_item(rng, final_item)
        sb.put_random(old_item)   # actor present here
        sb.leave(answerer)        # actor leaves (now uncertain)
        # To make the belief “not guaranteed to be correct”, perform a flip after they leave.
        sb.put_random(final_item)

    else:
        # Fallback: ensure final truth is narrated
        sb.put_random(final_item)


def _compute_minimal_present_initially(sb: _Builder,
                                       must_be_present_end: Set[str]) -> List[str]:
    """
    Minimal initial presence:
      - anybody who acted or left (sb.used)
      - anybody required to be present at end (must_be_present_end)
    Anyone else is omitted.
    """
    initial = set(sb.used) | set(must_be_present_end)
    # Also ensure that anyone who appears in any event (including as answerer in a leave) starts present.
    # sb.used already includes all actors of puts/moves/leaves.
    return sorted(initial)


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


def generate_scenarios_from_tuples(specs: List[SpecTuple], seed: Optional[int] = None) -> List['Scenario']:
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
      - PLAYER_HAS_UNCERTAINTY: actor forms a belief, leaves, and then we flip the asked container after they leave (so belief is not guaranteed to be correct).
      - AskConstraint for PLAYER_*: we remove higher-priority targets by never including them and ensure the intended target is present to witness final truth (when applicable).
    """
    rng = random.Random(seed)
    scenarios: List[Scenario] = []

    for i, (ent, ask, actor_role) in enumerate(specs):
        actor = _actor_name_from_role(actor_role)
        qc = rng.choice(CONTAINERS_GEN)
        answerer = _select_who_answers(actor, ent, rng)

        # Plan availability (who can be present initially)
        available: Set[str] = {'A', 'B', 'C', 'D', 'N'}

        intended_knower: Optional[str] = None
        must_be_present_end: Set[str] = set()

        # Enforce AskConstraint only for PLAYER_* ontologies
        if ent in (EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF):
            available, intended_knower = _plan_availability_for_player_ask(actor, ask)

        # Ontology-specific availability tweaks for the answerer
        # - False belief: answerer must be present initially to witness initial put
        # - No belief: answerer can be absent entirely (we prefer "never present" to keep minimality)
        if ent in (EpistemicType.TEAMMATE_HAS_FALSE_BELIEF, EpistemicType.OPPONENT_HAS_FALSE_BELIEF,
                   EpistemicType.PLAYER_HAS_UNCERTAINTY):
            available.add(answerer)
        elif ent in (EpistemicType.TEAMMATE_HAS_NO_BELIEF, EpistemicType.PLAYER_HAS_NO_BELIEF):
            # If answerer is in 'available', we'll schedule an immediate leave; otherwise, they were never present.
            pass
        elif ent in (EpistemicType.TEAMMATE_HAS_TRUE_BELIEF, EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY,
                     EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY,
                     EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY):
            available.add(answerer)

        # Ensure intended knower (for AskConstraint) is allowed to be present
        if intended_knower:
            available.add(intended_knower)
            # If the policy is going to ask this person, we also want them present at end
            must_be_present_end.add(intended_knower)

        # Certainty semantics: for ...WITH_CERTAINTY, answerer must be present through end
        if ent == EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY:
            must_be_present_end.add(answerer)

        # Build events with a presence-safe builder
        sb = _Builder(rng, qc, available)
        final_item = rng.choice(ITEMS_GEN)

        # Build belief state for the answerer
        _build_answerer_belief(sb, ent, answerer, final_item, rng)

        # If this is a PLAYER_* Ask scenario, ensure the intended knower witnesses the final truth
        if ent in (EpistemicType.PLAYER_HAS_UNCERTAINTY, EpistemicType.PLAYER_HAS_NO_BELIEF):
            _ensure_intended_knower_witnesses(sb, intended_knower, final_item)

        # Minimal initial presence
        present_initially = _compute_minimal_present_initially(sb, must_be_present_end)
        if actor not in present_initially:
            present_initially.append(actor)
            present_initially.sort()

        witness_required = {
            EpistemicType.TEAMMATE_HAS_TRUE_BELIEF,
            EpistemicType.OPPONENT_HAS_TRUE_BELIEF_WITH_CERTAINTY,
            EpistemicType.HONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY,
            EpistemicType.DISHONEST_OPPONENT_HAS_TRUE_BELIEF_WITH_UNCERTAINTY,
        }
        if ent in witness_required and answerer not in present_initially:
            present_initially.append(answerer)
            present_initially.sort()

        scenario = Scenario(
            round_num=(i // 4) + 1,
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

    return scenarios


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
    scenarios = generate_scenarios_from_tuples(specs, seed=None)
    outfile = 'scenarios_generated2.json'
    save_scenarios(scenarios, outfile)
    print(f"Created {outfile} with auto-generated scenarios")
