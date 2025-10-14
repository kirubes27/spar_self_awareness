# Minimal Flask wrapper around tom_test.py
from __future__ import annotations
import os
import tempfile
import uuid
from typing import Dict, Any, Optional, Tuple

from flask import Flask, render_template, request, jsonify, make_response

import tom_test as rg  # your existing game code (unchanged)

app = Flask(__name__)

# In-memory sessions (simple and fast for local use)
SESSIONS: Dict[str, "Session"] = {}

CLI_ACTION_PROMPT = "Your action (Ask(Player, Container), Tell(Player, Container, Contents), or Pass):"
PRESS_ENTER = "[Press Enter to continue]"

def fmt_delta(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}{abs(x)}"

def pick_scenarios_file() -> Optional[str]:
    if os.path.exists("scenarios_generated2.json"):
        return "scenarios_generated2.json"
    if os.path.exists("scenarios.json"):
        return "scenarios.json"
    return None

def initial_transcript() -> str:
    # Instructions header exactly like CLI
    path = pick_scenarios_file()
    try:
        scenarios = rg.load_scenarios(path) if path else rg.get_default_scenarios()
    except Exception:
        scenarios = rg.get_default_scenarios()
    win_score = max(rg.WINNING_SCORE, int(len(scenarios) / 4) + 4)
    return "=" * 70 + "\n" + rg.GAME_SETUP.format(WINNING_SCORE=win_score) + "\n" + "=" * 70 + "\n"

def present_scenario(state: Dict[str, Any], transcript: str) -> Tuple[str, Dict[str, Any], bool, bool]:
    game = state["game"]
    scenario = game.get_current_scenario()
    turn_char = game.get_current_turn_character()
    if not scenario or not turn_char:
        transcript = finalize_game(transcript, state)
        return transcript, state, False, True

    true_contents = game.process_scenario_events(scenario)
    state["true_contents"] = true_contents

    turn_count = state["turn_count"]
    transcript += "\n***********************************\n"
    transcript += f"Score: Blue={game.scores[rg.Team.BLUE]}, Red={game.scores[rg.Team.RED]}\n"
    transcript += ("Here's the first scenario:\n" if turn_count == 0 else f"Here's scenario {turn_count + 1}:\n")

    turn_name = "Your" if turn_char == 'A' else f"{turn_char}'s"
    transcript += f"{turn_name} turn:\n"

    scenario_desc = scenario.get_description_for(turn_char, game.characters)
    answerer = "you" if scenario.who_answers == turn_char else scenario.who_answers
    question_desc = f"I am going to ask {answerer} what is in the {scenario.question_container}."

    transcript += "-----------------------------------------------\n"
    transcript += scenario_desc + "\n"
    transcript += "----------------------------------------------\n"
    transcript += question_desc + "\n"
    transcript += "Respond ONLY with your action, and no other text.\n"

    state["last_scenario_desc"] = scenario_desc
    state["last_question_desc"] = question_desc
    return transcript, state, (turn_char == 'A'), False

def resolve_turn(state: Dict[str, Any], actor_name: str, action: rg.Action, transcript: str) -> Tuple[str, Dict[str, Any]]:
    game = state["game"]
    scenario = game.get_current_scenario()
    true_contents = state["true_contents"]

    score_delta, _ = game.execute_action(actor_name, action, true_contents)

    if action.action_type == rg.ActionType.ASK:
        action_str = f"Ask({action.target_char}, {action.container})"
    elif action.action_type == rg.ActionType.TELL:
        action_str = f"Tell({action.target_char}, {action.container}, {action.contents})"
    else:
        action_str = "Pass"
    transcript += f"\nAction: {action_str}\n"

    answer_given, is_correct, answer_score = game.resolve_answer_phase(scenario, true_contents)
    transcript += f"{scenario.who_answers} answers: {answer_given}\n"
    if is_correct:
        transcript += f"Correct! The {scenario.question_container} contains {answer_given}.\n"
    else:
        transcript += f"Incorrect. The {scenario.question_container} actually contains {true_contents[scenario.question_container]}.\n"

    blue_delta = 0.0
    red_delta = 0.0
    if actor_name in ['A', 'B']:
        blue_delta += score_delta
    else:
        red_delta += score_delta
    if is_correct:
        if scenario.who_answers in ['A', 'B']:
            blue_delta += answer_score
        else:
            red_delta += answer_score

    game.scores[rg.Team.BLUE] += blue_delta
    game.scores[rg.Team.RED] += red_delta
    transcript += f"\nOutcome: Blue {fmt_delta(blue_delta)}, Red {fmt_delta(red_delta)}\n"

    # Record, exactly like CLI
    if actor_name == 'A':
        was_optimal = game.is_action_optimal(action_str, scenario, true_contents)
        exp = game.execute_npc_action(actor_name, scenario, true_contents)
        if exp.action_type == rg.ActionType.PASS:
            expected_action_str = "Pass"
        elif exp.action_type == rg.ActionType.ASK:
            expected_action_str = f"Ask({exp.target_char}, {exp.container})"
        else:
            expected_action_str = f"Tell({exp.target_char}, {exp.container}, {exp.contents})"
    else:
        was_optimal = True
        expected_action_str = action_str

    game.turn_records.append(rg.TurnRecord(
        round_num=scenario.round_num,
        character=actor_name,
        scenario_desc=state.get("last_scenario_desc", ""),
        question=state.get("last_question_desc", ""),
        action=action_str,
        action_cost=abs(score_delta),
        answer_given=answer_given,
        answer_correct=is_correct,
        answer_score=answer_score,
        optimal_action=expected_action_str,
        was_optimal=was_optimal,
        blue_score_after=game.scores[rg.Team.BLUE],
        red_score_after=game.scores[rg.Team.RED],
        epistemic_type=scenario.epistemic_type.value if scenario.epistemic_type else None,
        ask_constraint=scenario.ask_constraint.value if scenario.ask_constraint else None
    ))
    return transcript, state

def finalize_game(transcript: str, state: Dict[str, Any]) -> str:
    game = state["game"]
    transcript += "\n" + "=" * 70 + "\n"
    transcript += "GAME OVER\n"
    transcript += f"Final Score: Blue {game.scores[rg.Team.BLUE]} - Red {game.scores[rg.Team.RED]}\n"
    if game.scores[rg.Team.BLUE] > game.scores[rg.Team.RED]:
        winner = rg.Team.BLUE
    elif game.scores[rg.Team.RED] > game.scores[rg.Team.BLUE]:
        winner = rg.Team.RED
    else:
        winner = None
    transcript += (f"Winner: {winner.value} team\n" if winner else "It's a tie!\n")
    transcript += "=" * 70 + "\n"

    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "game_results.json")
    rg.save_game_results(game.turn_records, out_path)
    state["results_path"] = out_path
    transcript += "\nGame results saved to game_results.json (download below)\n"
    return transcript

class Session:
    def __init__(self):
        self.transcript = initial_transcript()
        self.state: Optional[Dict[str, Any]] = None
        self.mode: str = "awaiting_start"  # awaiting_start | awaiting_action | awaiting_continue | over
        self.primary_label = "Start"
        self.placeholder = "[Press Enter to start]"

    def start_or_continue(self):
        if self.state is None:
            # Start game
            path = pick_scenarios_file()
            game = rg.create_game(path)
            rg.WINNING_SCORE = max(rg.WINNING_SCORE, int(len(game.scenarios) / len(game.turn_order)) + 4)
            self.state = {"game": game, "turn_count": 0, "results_path": None}
            self.transcript, self.state, is_player, end_now = present_scenario(self.state, self.transcript)
            if end_now:
                self.transcript = finalize_game(self.transcript, self.state)
                self.mode = "over"
                self.primary_label = "Start"
                self.placeholder = "Game over"
                return
            if is_player:
                self.mode = "awaiting_action"
                self.primary_label = "Continue"
                self.placeholder = CLI_ACTION_PROMPT
                return
            # NPC acts then pause
            npc = game.execute_npc_action(game.get_current_turn_character(), game.get_current_scenario(), self.state["true_contents"])
            self.transcript, self.state = resolve_turn(self.state, game.get_current_turn_character(), npc, self.transcript)
            game.check_game_over(); game.advance_turn(); self.state["turn_count"] += 1
            if game.game_over or not game.get_current_scenario():
                self.transcript = finalize_game(self.transcript, self.state)
                self.mode = "over"
                self.primary_label = "Start"
                self.placeholder = "Game over"
                return
            self.mode = "awaiting_continue"
            self.primary_label = "Continue"
            self.placeholder = PRESS_ENTER
            self.transcript += f"\n{PRESS_ENTER}\n"
            return

        # Continue flow
        if self.mode != "awaiting_continue":
            return
        game = self.state["game"]
        self.transcript, self.state, is_player, end_now = present_scenario(self.state, self.transcript)
        if end_now:
            self.transcript = finalize_game(self.transcript, self.state)
            self.mode = "over"
            self.primary_label = "Start"
            self.placeholder = "Game over"
            return
        if is_player:
            self.mode = "awaiting_action"
            self.primary_label = "Continue"
            self.placeholder = CLI_ACTION_PROMPT
            return
        # NPC acts then pause
        npc = game.execute_npc_action(game.get_current_turn_character(), game.get_current_scenario(), self.state["true_contents"])
        self.transcript, self.state = resolve_turn(self.state, game.get_current_turn_character(), npc, self.transcript)
        game.check_game_over(); game.advance_turn(); self.state["turn_count"] += 1
        if game.game_over or not game.get_current_scenario():
            self.transcript = finalize_game(self.transcript, self.state)
            self.mode = "over"
            self.primary_label = "Start"
            self.placeholder = "Game over"
            return
        self.mode = "awaiting_continue"
        self.primary_label = "Continue"
        self.placeholder = PRESS_ENTER
        self.transcript += f"\n{PRESS_ENTER}\n"

    def submit_action(self, text: str):
        if self.state is None or self.mode != "awaiting_action":
            return
        game = self.state["game"]
        action = game.parse_action(text or "")
        if not action:
            self.transcript += "Invalid action format. Try again.\n"
            self.placeholder = CLI_ACTION_PROMPT
            return
        turn_char = game.get_current_turn_character()
        self.transcript, self.state = resolve_turn(self.state, turn_char, action, self.transcript)
        game.check_game_over(); game.advance_turn(); self.state["turn_count"] += 1
        if game.game_over or not game.get_current_scenario():
            self.transcript = finalize_game(self.transcript, self.state)
            self.mode = "over"
            self.primary_label = "Start"
            self.placeholder = "Game over"
            return
        self.mode = "awaiting_continue"
        self.primary_label = "Continue"
        self.placeholder = PRESS_ENTER
        self.transcript += f"\n{PRESS_ENTER}\n"

def get_session():
    sid = request.cookies.get("sid")
    set_cookie = False
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = Session()
        set_cookie = True
    return sid, SESSIONS[sid], set_cookie

@app.get("/")
def index():
    sid, sess, set_cookie = get_session()
    resp = make_response(render_template("index.html"))
    if set_cookie:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

@app.get("/state")
def state():
    sid, sess, set_cookie = get_session()
    resp = jsonify({
        "transcript": sess.transcript,
        "mode": sess.mode,
        "primary_label": sess.primary_label,
        "placeholder": sess.placeholder
    })
    if set_cookie:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

@app.post("/primary")
def primary():
    sid, sess, set_cookie = get_session()
    sess.start_or_continue()
    resp = jsonify({
        "transcript": sess.transcript,
        "mode": sess.mode,
        "primary_label": sess.primary_label,
        "placeholder": sess.placeholder
    })
    if set_cookie:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

@app.post("/action")
def action():
    sid, sess, set_cookie = get_session()
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    sess.submit_action(text)
    resp = jsonify({
        "transcript": sess.transcript,
        "mode": sess.mode,
        "primary_label": sess.primary_label,
        "placeholder": sess.placeholder
    })
    if set_cookie:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

if __name__ == "__main__":
    import os
    debug = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)