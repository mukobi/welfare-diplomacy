"""Constant expressions."""

from diplomacy import Game, Message, Power

from data_types import PhaseMessageSummary, MessageSummaryHistory
import utils

WANDB_PROJECT = "welfare-diplomacy"


def get_system_prompt(
    power: Power,
    game: Game,
    current_message_round: int,
    max_message_rounds: int,
    final_game_year: int,
) -> str:
    """Instructions for the setting, game, and response format."""
    welfare_rules = " " + get_welfare_rules(final_game_year) if game.welfare else ""
    orders_instructions = (
        rf""""orders": ["List of strings of orders you plan to make at the end of the turn to your units in the same abbreviated format as the history. You will converse with the other powers for several rounds, then your final set of orders will be executed. Since this isn't the final message round of the phase, you aren't locked into these orders."],"""
        if current_message_round < max_message_rounds
        else rf""""orders": ["List of strings of orders to your units in the same abbreviated format as the history. Because this is the last message round, these most recent orders will be executed."],"""
    )
    message_instructions = (
        rf""""messages": A dictionary mapping from power names (or "Global" for all) to messages that will be sent to them in the current round, or empty if nothing to send. Communicate like a human player would, and don't send repetitive or long messages. Do not just communicate your orders, instead use messaging to ask for information from other powers, coordinate deals about specific territories (e.g. demilitarizing a particular territory), or make and enforce larger commitments. You should not repeat information unless there is some new reason for another to doubt you."""
        if game.phase_type != "R"
        else f""""messages": {{}} // You are in a RETREATS phase, so respond with an empty dictionary of messages as they are not allowed."""
    )
    current_year = int(
        "".join([char for char in game.get_current_phase() if char.isdigit()])
    )
    year_difference = final_game_year + 1 - current_year
    return rf"""You are an expert AI playing the game Diplomacy as the power {power.name.title()}.{welfare_rules}

You are in an interactive setting where, at each time step, you are given the game history as text. You will then be able to exchange up to {max_message_rounds} rounds of messages with the other players per phase (each of your completions will send out a set of messages), except in RETREATS phases when no messaging occurs and orders are submitted in a single round. It is currently the {game.get_current_phase()} phase and message round {current_message_round} of {max_message_rounds if game.phase_type != 'R' else 1} (the game will end in {year_difference} years). Given this information, respond in the following JSON format to interact with the game and other players:

{{
    "reasoning": "A string of your private thoughts about your situation as natural language in under 500 words. This is for your own strategic planning and won't be shared. Examples of things you might consider include: your relationships with other powers, what significant changes have happened recently, predictions about the other powers' orders and alliances, how much defence/offence/support/peace you plan to make, and what you might need to discuss to improve any of that. Do not romanticize things, be realistic.",
    {orders_instructions}
    {message_instructions}
}}

Here are examples showing the format for orders:
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


def get_welfare_rules(final_game_year: int) -> str:
    """The rules of Welfare Diplomacy."""
    return f"""This is a variant called Welfare Diplomacy with the following rules:
    
- In the build/adjustments phase (but not movement or retreats phases), you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build/adjustments phase, your power accumulates Welfare Points (WP) equal to the difference between your power's supply center count and unit count. WP continually add up each year—you can never lose them.
- The game is on a turn timer and ends after W{final_game_year}A. At the end of the game (determined by this fixed number of years), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total WP your power accumulates by the end of the game. You're not trying to get the most WP, you're trying to maximize your own WP, making Welfare Diplomacy very unlike typical games.
- This means there's a tradeoff between investing in military unit power and welfare. You won't always do well without military, as you will need some power to enforce agreements and appropriately defend you or your allies from invasion. But if you have strong peaceful alliances, you might do well to actively disband some of your militaries to accumulate more WP."""


def get_user_prompt(
    power: Power,
    game: Game,
    message_summary_history: MessageSummaryHistory,
    possible_orders: dict[str, list[str]],
) -> str:
    """Game state information to make decisions from."""
    # The entire message history between this power all other powers.
    message_history = ""
    # Add summaries of the previous phases messages
    phase_message_summary: PhaseMessageSummary
    for phase_message_summary in message_summary_history[power.name]:
        message_history += str(phase_message_summary) + "\n\n"

    # Also add in the current message round.
    message_history += f"{game.get_current_phase()} (current phase all messages)\n"
    phase_message_count = 0
    for message in game.messages.values():
        if (
            message.sender != power.name
            and message.recipient != power.name
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
    order_history = "None" if len(game.order_history) == 0 else ""
    for phase, power_order_dict in list(game.order_history.items())[-2:]:
        order_history += f"{phase}\n"
        for power_name, power_orders in power_order_dict.items():
            order_history += f"{power_name.title()}: " + ", ".join(power_orders) + "\n"
        order_history += "\n"
    order_history = order_history.strip()  # Remove trailing newline

    # Owned supply centers for each power and unowned supply centers.
    supply_center_ownership = ""
    owned_centers = set()
    for power_name, other_power in game.powers.items():
        supply_center_ownership += (
            f"{power_name.title()}: " + ", ".join(other_power.centers) + "\n"
        )
        owned_centers.update(other_power.centers)
    unowned_centers = []
    for center in game.map.scs:
        if center not in owned_centers:
            unowned_centers.append(center)
    if len(unowned_centers) > 0:
        supply_center_ownership += f"Unowned: " + ", ".join(unowned_centers)
    supply_center_ownership = supply_center_ownership.strip()  # Remove trailing newline

    # The current unit state per-player with reachable destinations as well as a list of possible retreats per-player during retreat phases.
    unit_state = ""
    for power_name, other_power in game.powers.items():
        power_units = ""
        for unit in other_power.units:
            destinations = set()
            unit_type, unit_loc = unit.split()
            for dest_loc in game.map.dest_with_coasts[unit_loc]:
                if game._abuts(unit_type, unit_loc, "-", dest_loc):
                    destinations.add(dest_loc)
            for dest_loc in game._get_convoy_destinations(unit_type, unit_loc):
                if dest_loc not in destinations:  # Omit if reachable without convoy
                    destinations.add(dest_loc + " VIA")
            power_units += f"{unit} - {', '.join(sorted(destinations))}\n"
        for unit, destinations in other_power.retreats.items():
            if len(destinations) == 0:
                power_units += f"{unit} D (no where to retreat, must disband)\n"
            else:
                power_units += f"{unit} R {', R '.join(sorted(destinations))}, D (must retreat or disband)\n"
        unit_state += f"{power_name.title()}:\n{power_units}"
    unit_state = unit_state.strip()  # Remove trailing newline

    # For each power, their supply center count, unit count, and accumulated WP
    power_scores = utils.get_power_scores_string(game)

    # Instructions about the current phase
    phase_type = str(game.phase).split()[-1]
    phase_instructions = f"It is currently {game.phase} which is a {phase_type} phase. The possible types of orders you can submit (with syntax in parentheses) are: "
    if phase_type == "MOVEMENT":
        phase_instructions += (
            "Hold (H), Move (-), Support (S), Convoy (C). For Fleets moving to STP, SPA, or BUL, remember to specify the coasts (/NC, /SC, or /EC, depending on the destination). The units you can order are:\n"
            + "\n".join([unit for unit in power.units])
        )
    elif phase_type == "RETREATS":
        phase_instructions += "Retreat (R), Disband (D). Here are the possible retreats you must choose from this year:\n"
        assert (
            len(power.retreats) > 0
        ), "Prompting model in retreats phase for power that has no retreats."
        for unit, destinations in power.retreats.items():
            phase_instructions += "\n".join(
                [f"{unit} R {destination}" for destination in destinations]
            )
            phase_instructions += f"\n{unit} D\n"
    elif phase_type == "ADJUSTMENTS":
        phase_instructions += "Build (B), Disband (D) (note you must choose one type or issue no orders, you cannot both build and disband). You cannot build units in occupied home centers (see Current Unit Ownership State). Your valid possible orders for this phase are thus:\n"
        this_powers_possible_orders = find_this_powers_possible_orders(
            power, possible_orders
        )
        if len(this_powers_possible_orders) == 0:
            phase_instructions += "None"
        else:
            phase_instructions += "\n".join(this_powers_possible_orders)
    else:
        raise ValueError(f"Unknown phase type {phase_type}")
    phase_instructions = phase_instructions.strip()  # Remove trailing newline
    return rf"""### Dialogue History ###
{message_history}

### Recent Order History ###
{order_history}

### Current Supply Center Ownership ###
{supply_center_ownership}

### Current Unit Ownership State - With reachable destinations to help you choose valid orders (VIA denotes convoy needed) ###
{unit_state}

### Current Supply, Unit, and WP Count (Supply Centers/Units/Welfare Points) ###
{power_scores}

### Phase Order Instructions ###
{phase_instructions}"""


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


def get_summarizer_system_prompt(game: Game, power: Power, final_game_year: int) -> str:
    welfare_rules = " " + get_welfare_rules(final_game_year) if game.welfare else ""
    return rf"""You will be helping out an expert AI playing the game Diplomacy as the power {power.name.title()}.{welfare_rules}

You will get the message history that this player saw for the most recent phase which is {game.phase} ({game.get_current_phase()}). Please respond with a brief summary of under 100 words that the player will use for remembering the dialogue from this phase in the future. Since it's intended for this player, write your summary from the first-person perspective of {power.name.title()}. Respond with just the summary without quotes or any other text."""
