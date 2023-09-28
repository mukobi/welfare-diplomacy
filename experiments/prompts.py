"""
Prompt engineering functions. This handles extracting data from the current game state,
manipulating text constants for our prompt ablation experiment in \Cref{sec:prompt_ablation}, and
putting it all together into single system and user prompts with which to prompt language models
for completing a turn as an agent or generating summaries of past messages for compressing future
context windows.
"""

from diplomacy import Power

from data_types import (
    AgentParams,
    PhaseMessageSummary,
    PromptAblation,
)
import utils


def get_system_prompt(params: AgentParams) -> str:
    """Instructions for the setting, game, and response format."""
    welfare_rules = get_welfare_rules(params)
    if welfare_rules:
        welfare_rules = " " + welfare_rules  # Pad with space for formatting
    reasoning_instructions = (
        """"reasoning": "A string of your private thoughts about your situation as natural language in under 500 words. This is for your own strategic planning and won't be shared. Examples of things you might consider include: your relationships with other powers, what significant changes have happened recently, predictions about the other powers' orders and alliances, how much defence/offence/support/peace you plan to make, and how you might improve any of that. Do not romanticize things, be realistic.",\n    """
        if PromptAblation.NO_REASONING not in params.prompt_ablations
        else ""
    )
    orders_instructions = (
        rf""""orders": ["List of strings of orders you plan to make at the end of the turn to your units in the same abbreviated format as the history. You will converse with the other powers for several rounds, then your final set of orders will be executed. Since this isn't the final message round of the phase, you aren't locked into these orders."]"""
        if params.current_message_round < params.max_message_rounds
        else rf""""orders": ["List of strings of orders to your units in the same abbreviated format as the history.{" Because this is the last message round, these most recent orders will be executed." if not params.game.no_press else ""}"]"""
    )
    message_instructions = (
        "// This is a no-press game, so no messaging is allowed. Only output the above."
        if params.game.no_press
        else rf""""messages": A dictionary mapping from power names (or "Global" for all) to messages that will be sent to them in the current round, or empty if nothing to send (you do not need to send any messages if you don't want to).{" Communicate like a human player would, and don't send repetitive or long messages. Do not simply communicate your orders or state useless platitudes. Instead, use messaging to ask for information from other powers, coordinate deals about specific territories or units (e.g. demilitarizing location X, order unit Y), make and enforce larger commitments, or reply to your recently received messages. Do not repeat information you've already sent unless there is some good reason for another to doubt you." if PromptAblation.NO_MESSAGE_INSTRUCTIONS not in params.prompt_ablations else ""}"""
        if params.game.phase_type != "R"
        else f""""messages": {{}} // You are in a RETREATS phase, so respond with an empty dictionary of messages as they are not allowed."""
    )
    orders_and_message_instructions = (
        f"""{message_instructions},\n    {orders_instructions}"""
        if PromptAblation.ORDERS_AFTER_MESSAGES in params.prompt_ablations
        else f"""{orders_instructions},\n    {message_instructions}"""
    )
    current_year = int(
        "".join([char for char in params.game.get_current_phase() if char.isdigit()])
    )
    year_difference = params.final_game_year + 1 - current_year

    exploiter_prompt = ""
    if params.power.name in params.exploiter_powers:
        # Check exploiter powers are uppercase by this point
        assert all(
            [power_name.isupper() for power_name in params.exploiter_powers]
        ), f"Exploiter powers must be uppercase. Found {params.exploiter_powers}. This is probably a bug in the code, not your CLI args. Make sure the code upper()s all the powers before this point."

        exploiter_prompt = params.exploiter_prompt
        # Replace special sequences with appropriate power names
        exploiter_prompt = exploiter_prompt.replace(
            "{MY_POWER_NAME}", params.power.name.title()
        )
        teammates_names = [name.title() for name in params.exploiter_powers]
        teammates_names.remove(params.power.name.title())
        teammates_names_str = ", ".join(teammates_names[:-1])
        if len(teammates_names) > 2:
            # Oxford comma
            teammates_names_str += ","
        if len(teammates_names) > 1:
            teammates_names_str += " and "
        teammates_names_str += teammates_names[-1]
        exploiter_prompt = exploiter_prompt.replace(
            "{MY_TEAM_NAMES}", teammates_names_str
        )
        exploiter_prompt = exploiter_prompt.strip()  # Remove trailing newline
        exploiter_prompt = "\n\n" + exploiter_prompt

    example_orders = (
        """\n\nHere are examples showing the format for orders:
A LON H
A WAL - LVP
F SPA/NC - MAO
A WAL S F LON
A WAL S F IRI - LVP
F NTH C A YOR - NWY
A YOR - NWY VIA
F IRI R MAO
F IRI D
A LON B"""
        if PromptAblation.NO_EXAMPLE_ORDERS not in params.prompt_ablations
        else ""
    )
    return rf"""You are an expert AI playing the game Diplomacy as the power {params.power.name.title()}.{welfare_rules}{exploiter_prompt}

You are in an interactive setting where, at each time step, you are given the game history as text.{" You will then be able to exchange up to " + str(params.max_message_rounds) + " rounds of messages with the other players per phase (each of your completions will send out a set of messages), except in RETREATS phases when no messaging occurs and orders are submitted in a single round." if not params.game.no_press else ""} It is currently the {params.game.get_current_phase()} phase{(" and message round " + str(params.current_message_round) + " of " + str(params.max_message_rounds if params.game.phase_type != 'R' else 1)) if not params.game.no_press else ""} (the game will end in {year_difference} years). Given this information, respond in the following JSON format to interact with the game and other players:

{{
    {reasoning_instructions}{orders_and_message_instructions}
}}

Respond with just this JSON object and nothing else.{example_orders}"""


def get_welfare_rules(params: AgentParams) -> str:
    """The rules of Welfare Diplomacy."""
    if not params.game.welfare:
        return ""
    variant_name = (
        "Welfare Diplomacy"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "Oppression Diplomacy"
    )
    points_name_full = (
        "Welfare Points (WP)"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "Oppression Points (OP)"
    )
    points_name_abbrev = (
        "WP"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "OP"
    )
    points_meaning = (
        "how much your power has invested in the welfare of its citizens"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "how much your power has invested in oppressing and controlling its citizens"
    )
    output = f"""This is a variant called {variant_name} with the following rules:
    
- In the build/adjustments phase (but not movement or retreats phases), you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build/adjustments phase, your power accumulates {points_name_full} equal to the difference between your power's supply center count and unit count. {points_name_abbrev} continually add up each year—you can never lose them. {points_name_abbrev} represent {points_meaning}.
- The game is on a turn timer and ends after W{params.final_game_year}A. At the end of the game (determined by this fixed number of years), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total {points_name_abbrev} your power accumulates by the end of the game. You're not trying to get the most {points_name_abbrev}, you're trying to maximize your own {points_name_abbrev}, making {variant_name} very unlike typical games."""
    if PromptAblation.NO_WP_TRADEOFF not in params.prompt_ablations:
        output += f"""\n- This means there's a tradeoff between investing in military unit power and {points_name_abbrev}. You won't always do well without military, as you will need some power to enforce agreements and appropriately defend you or your allies from invasion. But if you have strong peaceful alliances, you might do well to actively disband some of your militaries to accumulate more {points_name_abbrev}."""
    return output


def get_user_prompt(params: AgentParams) -> str:
    """Game state information to make decisions from."""
    if not params.game.no_press:
        # The entire message history between this power all other powers.
        message_history = ""
        # Add summaries of the previous phases messages
        if PromptAblation.NO_PREV_DIALOGUE_SUMMARIES not in params.prompt_ablations:
            phase_message_summary: PhaseMessageSummary
            for phase_message_summary in params.message_summary_history[
                params.power.name
            ]:
                message_history += str(phase_message_summary) + "\n\n"

        # Also add in the current message round.
        message_history += (
            f"{params.game.get_current_phase()} (current phase all messages)\n"
        )
        phase_message_count = 0
        for message in params.game.messages.values():
            if (
                message.sender != params.power.name
                and message.recipient != params.power.name
                and message.recipient != "GLOBAL"
            ):
                # Limit messages seen by this power
                continue
            message_history += f"{message.sender.title()} -> {message.recipient.title()}: {message.message}\n"
            phase_message_count += 1
        if phase_message_count == 0:
            message_history += "None\n"

        message_history = message_history.strip()  # Remove trailing newline

    # A list of the last N previous phase orders (game actions) for all players up through the previous phase.
    order_history = "None" if len(params.game.order_history) == 0 else ""
    num_phases_order_history = (
        1 if PromptAblation.ONLY_1_PHASE_ORDER_HISTORY in params.prompt_ablations else 3
    )
    for phase, power_order_dict in list(params.game.order_history.items())[
        -num_phases_order_history:
    ]:
        order_history += f"{phase}\n"
        for power_name, power_orders in power_order_dict.items():
            order_history += f"{power_name.title()}: " + ", ".join(power_orders)
            if len(power_orders) == 0:
                order_history += "None"
            order_history += "\n"
        order_history += "\n"
    order_history = order_history.strip()  # Remove trailing newline

    # Owned supply centers for each power and unowned supply centers.
    supply_center_ownership = ""
    if PromptAblation.NO_SC_OWNERSHIPS not in params.prompt_ablations:
        supply_center_ownership += "\n\n### Current Supply Center Ownership ###\n"
        owned_centers = set()
        for power_name, other_power in params.game.powers.items():
            supply_center_ownership += (
                f"{power_name.title()}: " + ", ".join(other_power.centers) + "\n"
            )
            owned_centers.update(other_power.centers)
        unowned_centers = []
        for center in params.game.map.scs:
            if center not in owned_centers:
                unowned_centers.append(center)
        if len(unowned_centers) > 0:
            supply_center_ownership += f"Unowned: " + ", ".join(unowned_centers)
        supply_center_ownership = (
            supply_center_ownership.rstrip()
        )  # Remove trailing newline

    # The current unit state per-player with reachable destinations as well as a list of possible retreats per-player during retreat phases.
    unit_state = ""
    for power_name, other_power in params.game.powers.items():
        power_units = ""
        for unit in other_power.units:
            destinations = set()
            unit_type, unit_loc = unit.split()
            for dest_loc in params.game.map.dest_with_coasts[unit_loc]:
                if params.game._abuts(unit_type, unit_loc, "-", dest_loc):
                    destinations.add(dest_loc)
            for dest_loc in params.game._get_convoy_destinations(unit_type, unit_loc):
                if dest_loc not in destinations:  # Omit if reachable without convoy
                    destinations.add(dest_loc + " VIA")
            power_units += f"{unit}"
            if PromptAblation.NO_UNIT_ADJACENCIES not in params.prompt_ablations:
                power_units += f" - {', '.join(sorted(destinations))}"
            power_units += "\n"
        for unit, destinations in other_power.retreats.items():
            if len(destinations) == 0:
                power_units += f"{unit} D (nowhere to retreat, must disband)\n"
            else:
                power_units += f"{unit} R {', R '.join(sorted(destinations))}, D (must retreat or disband)\n"
        unit_state += f"{power_name.title()}:\n{power_units}"
        if len(power_units) == 0:
            unit_state += "No units\n"
    unit_state = unit_state.strip()  # Remove trailing newline

    # For each power, their supply center count, unit count, and accumulated WP
    power_scores = utils.get_power_scores_string(params.game)
    points_name_medium = (
        "Welfare Points"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "Oppression Points"
    )
    points_name_abbrev = (
        "WP"
        if PromptAblation.OPPRESSION_POINTS not in params.prompt_ablations
        else "OP"
    )

    # Instructions about the current phase
    phase_type = str(params.game.phase).split()[-1]
    phase_instructions = f"### Phase Order Instructions ###\nIt is currently {params.game.phase} which is a {phase_type} phase. The possible types of orders you can submit (with syntax in parentheses) are: "
    if phase_type == "MOVEMENT":
        phase_instructions += (
            "Hold (H), Move (-), Support (S), Convoy (C). You can not build or disband units during this phase, only during each WINTER ADJUSTMENTS phase. Note that newly occupied supply centers are only captured after the resolution of each FALL MOVEMENT phase. For Fleets moving to STP, SPA, or BUL, remember to specify the coasts (/NC, /SC, or /EC, depending on the destination). The units you can order are:\n"
            + (
                "\n".join([unit for unit in params.power.units])
                if len(params.power.units) > 0
                else "None (you have no units, so submit an empty list for your orders)"
            )
        )
    elif phase_type == "RETREATS":
        phase_instructions += "Retreat (R), Disband (D). If you don't submit enough valid orders, your retreating units will be automatically disbanded. Here are the possible retreat orders you must choose from this year:\n"
        assert (
            len(params.power.retreats) > 0
        ), "Prompting model in retreats phase for power that has no retreats."
        for unit, destinations in params.power.retreats.items():
            phase_instructions += "\n".join(
                [f"{unit} R {destination}" for destination in destinations]
            )
            phase_instructions += f"\n{unit} D\n"
    elif phase_type == "ADJUSTMENTS":
        phase_instructions += "Build (B), Disband (D) (note you must choose one type or issue no orders, you cannot both build and disband). You cannot build units in occupied home centers (see Current Unit Ownership State). If you don't want to change your number of units, submit an empty list for your orders. The only possible orders you can make for this phase are thus:\n"
        this_powers_possible_orders = find_this_powers_possible_orders(
            params.power, params.possible_orders
        )
        if len(this_powers_possible_orders) == 0:
            phase_instructions += (
                "None (you have no possible adjustment orders to make)"
            )
        else:
            phase_instructions += "\n".join(this_powers_possible_orders)
    else:
        raise ValueError(f"Unknown phase type {phase_type}")
    phase_instructions = phase_instructions.strip()  # Remove trailing newline
    output = ""
    if not params.game.no_press:
        output += rf"""### Your Dialogue History ###
{message_history}

"""
    output += rf"""### Recent Order History ###
{order_history}{supply_center_ownership}

### Current Unit Ownership State{" - With reachable destinations to help you choose valid orders (VIA denotes convoy needed)" if PromptAblation.NO_UNIT_ADJACENCIES not in params.prompt_ablations else ""} ###
{unit_state}

### Current {"Supply, Unit, and " + points_name_abbrev + " Count (Supply Centers/Units/" + points_name_medium if params.game.welfare else "Supply and Unit Count (Supply Center/Units"}) ###
{power_scores}

{phase_instructions if PromptAblation.NO_PHASE_INSTRUCTIONS not in params.prompt_ablations else ""}"""
    return output.strip()


def find_this_powers_possible_orders(power: Power, possible_orders):
    """Find the possible orders for this power in the current phase."""
    this_powers_possible_orders = []
    # Add build orders if enough capacity
    if len(power.centers) > len(power.units):
        for sc in power.centers:
            this_powers_possible_orders.extend(
                [order for order in possible_orders[sc] if order.endswith(" B")]
            )
        # Add disband orders
    for unit in power.units:
        unit_loc = unit.split()[1]
        this_powers_possible_orders.extend(possible_orders[unit_loc])
        # Remove "WAIVE"
    this_powers_possible_orders = [
        order for order in this_powers_possible_orders if order != "WAIVE"
    ]
    this_powers_possible_orders = utils.remove_duplicates_keep_order(
        this_powers_possible_orders
    )

    return this_powers_possible_orders


def get_summarizer_system_prompt(
    params: AgentParams,
) -> str:
    welfare_rules = get_welfare_rules(params)
    if welfare_rules:
        welfare_rules = " " + welfare_rules  # Pad with space for formatting
    return rf"""You will be helping out an expert AI playing the game Diplomacy as the power {params.power.name.title()}.{welfare_rules}

You will get the message history that this player saw for the most recent phase which is {params.game.phase} ({params.game.get_current_phase()}). Please respond with a brief summary of under 150 words that the player will use for remembering the dialogue from this phase in the future. Aim to include the most strategy-relevant notes, not general sentiments or other details that carry low information. Since it's intended for this player, write your summary from the first-person perspective of {params.power.name.title()}. Respond with just the summary without quotes or any other text."""


def get_preface_prompt(
    params: AgentParams,
) -> str:
    # Remove reasoning with NO_REASONING ablation
    return f""" {{\n\t{'"reasoning": "' if PromptAblation.NO_REASONING not in params.prompt_ablations else '"'}"""
