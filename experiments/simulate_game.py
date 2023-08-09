"""
Language model scaffolding to play Diplomacy.


"""

import argparse
import logging
import os

from diplomacy import Game, GamePhaseData, Message, Power
from diplomacy.utils.export import to_saved_game_format
import numpy as np
from tqdm import tqdm
import wandb
from wandb.integration.openai import autolog

from agents import Agent, AgentCompletionError, model_name_to_agent
from data_types import AgentResponse, MessageSummaryHistory
from message_summarizers import (
    MessageSummarizer,
    model_name_to_message_summarizer,
)
import prompts
import utils


def main():
    """Simulate a game of Diplomacy with the given parameters."""
    # Parse args
    args = parse_args()

    # Initialize seed, wandb, game, logger, agent, and summarizer
    wandb.init(
        entity=args.entity,
        project=args.project,
        save_code=True,
        config=vars(args),
        mode="disabled" if args.disable_wandb else "online",
        settings=wandb.Settings(code_dir="experiments"),
    )
    assert wandb.run is not None
    autolog()  # Logs OpenAI API calls to wandb

    utils.set_seed(wandb.config.seed)

    game: Game = Game(map_name=wandb.config.map_name)
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(wandb.config.log_level)

    agent: Agent = model_name_to_agent(
        wandb.config.agent_model,
        temperature=wandb.config.temperature,
        top_p=wandb.config.top_p,
    )
    message_summarizer: MessageSummarizer = model_name_to_message_summarizer(
        wandb.config.summarizer_model, logger=logger
    )
    message_summary_history: MessageSummaryHistory = MessageSummaryHistory()
    for power_name in game.powers.keys():
        message_summary_history[power_name] = []

    utils.log_info(
        logger,
        f"Starting game with map {wandb.config.map_name} and agent model {wandb.config.agent_model} summarized by {message_summarizer} ending after {wandb.config.max_years} years with {wandb.config.max_message_rounds} message rounds per phase.",
    )

    # Log the initial state of the game
    rendered_with_orders = game.render(incl_abbrev=True)
    log_object = {
        "meta/year_fractional": 0.0,
        "board/rendering_with_orders": wandb.Html(rendered_with_orders),
        "board/rendering_state": wandb.Html(rendered_with_orders),
    }
    for power in game.powers.values():
        short_name = power.name[:3]
        log_object[f"score/units/{short_name}"] = len(power.units)
        log_object[f"score/welfare/{short_name}"] = power.welfare_points
        log_object[f"score/centers/{short_name}"] = len(power.centers)

    welfare_list = [power.welfare_points for power in game.powers.values()]
    log_object["welfare/hist"] = wandb.Histogram(welfare_list)
    log_object["welfare/min"] = np.min(welfare_list)
    log_object["welfare/max"] = np.max(welfare_list)
    log_object["welfare/mean"] = np.mean(welfare_list)
    log_object["welfare/median"] = np.median(welfare_list)
    log_object["welfare/total"] = np.sum(welfare_list)

    wandb.log(log_object)

    simulation_max_years = (
        wandb.config.early_stop_max_years
        if wandb.config.early_stop_max_years > 0
        else wandb.config.max_years
    )
    final_game_year = wandb.config.max_years + 1900

    # Initialize global counters
    all_num_conflicts: list[int] = []
    all_num_holds: list[int] = []
    all_num_moves: list[int] = []
    all_num_supports: list[int] = []
    all_num_convoys: list[int] = []
    all_num_builds: list[int] = []
    all_num_disbands: list[int] = []
    all_ratio_moves: list[float] = []
    all_ratio_supports: list[float] = []
    all_ratio_builds: list[float] = []
    all_num_centers_lost: list[int] = []
    all_valid_ratio_averages: list[float] = []
    all_num_completion_errors: list[int] = []

    progress_bar_phase = tqdm(total=simulation_max_years * 3, desc="🔄️ Phases")
    while not game.is_game_done:
        utils.log_info(logger, f"🕰️  Beginning phase {game.get_current_phase()}")

        # Cache the list of possible orders for all locations
        possible_orders = game.get_all_possible_orders()

        total_num_orders = 0
        total_num_valid_orders = 0
        list_valid_order_ratios = []
        phase_num_completion_errors = 0
        total_message_sent = 0
        # (power_name, message_round, agent_response, invalid orders)
        agent_response_history: list[tuple[str, int, AgentResponse, list[str]]] = []
        list_completion_times_sec = []
        list_prompt_tokens = []
        list_completion_tokens = []
        list_total_tokens = []
        message_history: list[tuple(str, int, str, str, str)] = []

        # During Retreats, only 1 round of completions without press
        num_of_message_rounds = (
            wandb.config.max_message_rounds if game.phase_type != "R" else 1
        )
        num_completing_powers = (
            len(game.powers)
            if game.phase_type != "R"
            else len([power for power in game.powers.values() if power.retreats])
        )
        progress_bar_messages = tqdm(
            total=num_of_message_rounds * num_completing_powers, desc="🙊 Messages"
        )
        for message_round in range(1, num_of_message_rounds + 1):
            # Randomize order of powers
            powers_items = list(game.powers.items())
            np.random.shuffle(powers_items)

            utils.log_info(
                logger,
                f"📨 Beginning message round {message_round}/{num_of_message_rounds}. Completion ordering: {', '.join([name for name, _ in powers_items])}",
            )

            count_completions_one_round = 0
            power: Power
            for power_name, power in powers_items:
                # On retreat phases, skip powers that have no retreats to make
                if game.phase_type == "R" and not power.retreats:
                    continue

                # Prompting the model for a response
                try:
                    agent_response = agent.respond(
                        power,
                        game,
                        message_summary_history,
                        possible_orders,
                        message_round,
                        wandb.config.max_message_rounds,
                        final_game_year,
                    )
                except AgentCompletionError as exc:
                    # If the agent fails to complete, we need to log the error and continue
                    phase_num_completion_errors += 1
                    utils.log_error(
                        logger,
                        f"🚨 {power_name} {game.get_current_phase()} Round {message_round}: Agent {agent.model_name} failed to complete ({phase_num_completion_errors} errors this phase). Skipping. Exception:\n{exc}",
                    )
                    progress_bar_messages.update(1)
                    continue
                list_completion_times_sec.append(agent_response.completion_time_sec)
                list_prompt_tokens.append(agent_response.prompt_tokens)
                list_completion_tokens.append(agent_response.completion_tokens)
                list_total_tokens.append(agent_response.total_tokens)
                if game.phase_type == "R":
                    if len(agent_response.messages) > 0:
                        utils.log_warning(
                            logger, "No messages are allowed during retreats, clearing."
                        )
                        agent_response.messages = {}
                count_completions_one_round += 1
                utils.log_info(
                    logger,
                    f"⚙️  {power_name} {game.get_current_phase()} Round {message_round}: Agent {agent_response.model_name} took {agent_response.completion_time_sec:.2f}s to respond.\nReasoning: {agent_response.reasoning}\nOrders: {agent_response.orders}\nMessages: {agent_response.messages}",
                )

                # Check how many of the orders were valid
                num_valid_orders = 0
                invalid_orders = []
                for order in agent_response.orders:
                    if "WAIVE" in order or "VOID" in order:
                        utils.log_warning(
                            logger,
                            f"Order '{order}' should not be generated by agent",
                        )
                        num_valid_orders += 1
                        invalid_orders.append(order)
                        continue
                    word = order.split()
                    location = word[1]
                    if (
                        location in possible_orders
                        and order in possible_orders[location]
                    ):
                        num_valid_orders += 1
                    else:
                        invalid_orders.append(order)
                num_orders = len(agent_response.orders)
                valid_order_ratio = (
                    num_valid_orders / num_orders if num_orders > 0 else None
                )
                valid_order_display_percent = (
                    valid_order_ratio * 100.0
                    if valid_order_ratio is not None
                    else np.NaN
                )
                utils.log_info(
                    logger,
                    f"✔️  {power_name} valid orders: {num_valid_orders}/{num_orders} = {valid_order_display_percent:.2f}%",
                )
                total_num_orders += num_orders
                total_num_valid_orders += num_valid_orders
                if valid_order_ratio is not None:
                    list_valid_order_ratios.append(valid_order_ratio)

                agent_response_history.append(
                    (power_name, message_round, agent_response, invalid_orders)
                )

                # Set orders, clearing first due to multiple message rounds
                game.set_orders(power_name, [])
                game.set_orders(power_name, agent_response.orders)

                # Send messages
                for recipient, message in agent_response.messages.items():
                    game.add_message(
                        Message(
                            sender=power_name,
                            recipient=recipient,
                            message=message,
                            phase=game.get_current_phase(),
                        )
                    )
                    message_history.append(
                        (
                            game.get_current_phase(),
                            message_round,
                            power_name,
                            recipient,
                            message,
                        )
                    )
                    total_message_sent += 1

                progress_bar_messages.update(1)

        # Render saved orders and current turn message history before processing
        rendered_with_orders = game.render(incl_abbrev=True)
        messages_table = wandb.Table(
            columns=["phase", "round", "sender", "recipient", "message"],
            data=[
                [phase, round, sender, recipient, message]
                for (phase, round, sender, recipient, message) in message_history
            ],
        )

        # Save summaries of the message history
        for power_name, power in tqdm(
            game.powers.items(), desc="✍️ Summarizing messages"
        ):
            phase_message_summary = message_summarizer.summarize(
                game, power, final_game_year
            )
            message_summary_history[power_name].append(phase_message_summary)

        # Advance the game simulation to the next phase
        game.process()
        phase: GamePhaseData = game.get_phase_history()[-1]

        # Check whether to end the game
        if int(game.phase.split()[1]) - 1900 > simulation_max_years:
            game._finish([])

        # Compute things to log to Weights & Biases
        rendered_state = game.render(incl_abbrev=True)
        valid_ratio_average = (
            np.mean(list_valid_order_ratios)
            if len(list_valid_order_ratios) > 0
            else None
        )
        if valid_ratio_average is not None:
            all_valid_ratio_averages.append(valid_ratio_average)
        avg_game_valid_ratio_avg = (
            np.mean(all_valid_ratio_averages)
            if len(all_valid_ratio_averages) > 0
            else None
        )
        all_num_completion_errors.append(phase_num_completion_errors)
        phase_avg_num_completions = (
            total_message_sent / count_completions_one_round
            if count_completions_one_round > 0
            else None
        )
        model_response_table = wandb.Table(
            columns=[
                "phase",
                "power",
                "round",
                "model",
                "reasoning",
                "orders",
                "invalid_orders",
                "messages",
                "system_prompt",
                "user_prompt",
            ],
            data=[
                [
                    phase.name,
                    power_name,
                    response_message_round,
                    agent_response.model_name,
                    agent_response.reasoning,
                    agent_response.orders,
                    invalid_orders,
                    [
                        f"{power_name} -> {recipient}: {message}"
                        for recipient, message in agent_response.messages.items()
                    ],
                    agent_response.system_prompt,
                    agent_response.user_prompt,
                ]
                for power_name, response_message_round, agent_response, invalid_orders in agent_response_history
            ],
        )
        message_summary_table = wandb.Table(
            columns=[
                "phase",
                "power",
                "original_messages",
                "summary",
            ],
            data=[
                [
                    message_summaries[-1].phase,
                    power_name,
                    "\n".join(message_summaries[-1].original_messages),
                    message_summaries[-1].summary,
                ]
                for power_name, message_summaries in message_summary_history.items()
            ],
        )
        log_object = {
            "meta/year_fractional": utils.get_game_fractional_year(phase),
            "board/rendering_with_orders": wandb.Html(rendered_with_orders),
            "board/rendering_state": wandb.Html(rendered_state),
            "orders/num_total": total_num_orders,
            "orders/num_valid": total_num_valid_orders,
            "orders/avg_game_valid_ratio_avg": avg_game_valid_ratio_avg,
            "orders/phase_valid_ratio_avg": valid_ratio_average,
            f"orders/phase_valid_ratio_avg_phase_{game.phase_type}": valid_ratio_average,
            "messages/messages_table": messages_table,
            "messages/message_summary_table": message_summary_table,
            "messages/num_total": total_message_sent,
            "messages/num_avg": phase_avg_num_completions,
            "model/completion_time_sec_avg": np.mean(list_completion_times_sec),
            "model/response_table": model_response_table,
            "model/phase_num_completion_errors": phase_num_completion_errors,
            "model/avg_num_completion_errors": np.mean(all_num_completion_errors),
            "tokens/prompt_tokens_avg": np.mean(list_prompt_tokens)
            if list_prompt_tokens
            else None,
            "tokens/completion_tokens_avg": np.mean(list_completion_tokens)
            if list_completion_tokens
            else None,
            "tokens/total_tokens_avg": np.mean(list_total_tokens)
            if list_total_tokens
            else None,
            "tokens/prompt_tokens_min": np.min(list_prompt_tokens)
            if list_prompt_tokens
            else None,
            "tokens/completion_tokens_min": np.min(list_completion_tokens)
            if list_completion_tokens
            else None,
            "tokens/total_tokens_min": np.min(list_total_tokens)
            if list_total_tokens
            else None,
            "tokens/prompt_tokens_max": np.max(list_prompt_tokens)
            if list_prompt_tokens
            else None,
            "tokens/completion_tokens_max": np.max(list_completion_tokens)
            if list_completion_tokens
            else None,
            "tokens/total_tokens_max": np.max(list_total_tokens)
            if list_total_tokens
            else None,
            "tokens/prompt_tokens_median": np.median(list_prompt_tokens)
            if list_prompt_tokens
            else None,
            "tokens/completion_tokens_median": np.median(list_completion_tokens)
            if list_completion_tokens
            else None,
            "tokens/total_tokens_median": np.median(list_total_tokens)
            if list_total_tokens
            else None,
            "tokens/prompt_tokens_hist": wandb.Histogram(list_prompt_tokens),
            "tokens/completion_tokens_hist": wandb.Histogram(list_completion_tokens),
            "tokens/total_tokens_hist": wandb.Histogram(list_total_tokens),
        }

        for power in game.powers.values():
            short_name = power.name[:3]
            if phase.name[-1] == "A" or phase.name[-1] == "R":
                # Centers/welfare/units only change after adjustments or sometimes retreats
                log_object[f"score/units/{short_name}"] = len(power.units)
                log_object[f"score/welfare/{short_name}"] = power.welfare_points
                log_object[f"score/centers/{short_name}"] = len(power.centers)

        if phase.name[-1] == "A":
            # Aggregated welfare
            welfare_list = [power.welfare_points for power in game.powers.values()]
            log_object["welfare/hist"] = wandb.Histogram(welfare_list)
            log_object["welfare/min"] = np.min(welfare_list)
            log_object["welfare/max"] = np.max(welfare_list)
            log_object["welfare/mean"] = np.mean(welfare_list)
            log_object["welfare/median"] = np.median(welfare_list)
            log_object["welfare/total"] = np.sum(welfare_list)

            # Track builds and disbands
            num_builds = sum(
                sum(order[-1] == "B" for order in power_orders)
                for power_orders in phase.orders.values()
            )
            num_disbands = sum(
                sum(order[-1] == "D" for order in power_orders)
                for power_orders in phase.orders.values()
            )
            num_adjustments = sum(
                len(power_orders) for power_orders in phase.orders.values()
            )
            ratio_builds = num_builds / num_adjustments if num_adjustments > 0 else None
            all_num_builds.append(num_builds)
            all_num_disbands.append(num_disbands)
            if ratio_builds is not None:
                all_ratio_builds.append(ratio_builds)
            log_object["builds/phase_num_builds"] = num_builds
            log_object["builds/phase_num_disbands"] = num_disbands
            log_object["builds/avg_num_builds"] = np.mean(all_num_builds)
            log_object["builds/avg_num_disbands"] = np.mean(all_num_disbands)
            log_object["builds/avg_ratio_builds"] = np.mean(all_ratio_builds)

            # Conquest: number of centers lost
            num_centers_lost = len(game.lost)
            all_num_centers_lost.append(num_centers_lost)
            log_object["conquest/phase_num_centers_lost"] = num_centers_lost
            log_object["conquest/total_num_centers_lost"] = np.sum(all_num_centers_lost)
            log_object["conquest/avg_num_centers_lost"] = np.mean(all_num_centers_lost)

        if phase.name[-1] == "M":
            # Track combat as measured by number of tiles where multiple units moved or held
            combat_dicts = [moving_units for moving_units in game.combat.values()]
            num_moving_units = [sum(len(v) for v in d.values()) for d in combat_dicts]
            phase_num_conflicts = sum([count > 1 for count in num_moving_units])
            all_num_conflicts.append(phase_num_conflicts)
            log_object["combat/phase_num_conflicts"] = phase_num_conflicts
            log_object["combat/total_num_conflicts"] = np.sum(all_num_conflicts)
            log_object["combat/avg_num_conflicts"] = np.mean(all_num_conflicts)
            log_object["combat/min_num_conflicts"] = np.min(all_num_conflicts)
            log_object["combat/max_num_conflicts"] = np.max(all_num_conflicts)

            # Track commands to see ratios of different moves
            command_types = [command.split()[0] for command in game.command.values()]
            num_commands = len(command_types)
            num_hold = command_types.count("H")
            num_move = command_types.count("-")
            num_support = command_types.count("S")
            num_convoy = command_types.count("C")
            ratio_move = num_move / num_commands if num_commands > 0 else None
            ratio_support = num_support / num_commands if num_commands > 0 else None
            all_num_holds.append(num_hold)
            all_num_moves.append(num_move)
            all_num_supports.append(num_support)
            all_num_convoys.append(num_convoy)
            if ratio_move is not None:
                all_ratio_moves.append(ratio_move)
            if ratio_support is not None:
                all_ratio_supports.append(ratio_support)
            log_object["commands/phase_num_hold"] = num_hold
            log_object["commands/phase_num_move"] = num_move
            log_object["commands/phase_num_support"] = num_support
            log_object["commands/phase_num_convoy"] = num_convoy
            log_object["commands/phase_ratio_move"] = ratio_move
            log_object["commands/phase_ratio_support"] = ratio_support
            log_object["commands/avg_num_holds"] = np.mean(all_num_holds)
            log_object["commands/avg_num_moves"] = np.mean(all_num_moves)
            log_object["commands/avg_num_supports"] = np.mean(all_num_supports)
            log_object["commands/avg_num_convoys"] = np.mean(all_num_convoys)
            log_object["commands/avg_ratio_moves"] = np.mean(all_ratio_moves)
            log_object["commands/avg_ratio_supports"] = np.mean(all_ratio_supports)

        wandb.log(log_object)

        # Print some information about the game
        utils.log_info(
            logger, f"📊 {phase.name} {utils.get_power_scores_string(game, abbrev=True)}"
        )

        # Update the progress bar based on how many turns have progressed (just counting M and A)
        new_phase_type = game.phase_type
        if new_phase_type == "M":
            # Any to M, update 1
            progress_bar_phase.update(1)
        elif new_phase_type == "A":
            # M or R to A, update 1
            progress_bar_phase.update(1)
        elif new_phase_type == "R":
            # Retreats, don't count it
            pass
        else:
            raise ValueError(f"Unknown phase type {new_phase_type}")

    # Exporting the game to disk to visualize (game is appended to file)
    # Alternatively, we can do >> file.write(json.dumps(to_saved_game_format(game)))
    if wandb.config.save:
        if not os.path.exists(wandb.config.output_folder):
            os.makedirs(wandb.config.output_folder)
        output_id = "debug" if wandb.config.disable_wandb else wandb.run.id
        to_saved_game_format(
            game,
            output_path=os.path.join(
                wandb.config.output_folder, f"game-{output_id}.json"
            ),
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate a game of Diplomacy with the given parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level", dest="log_level", default="INFO", help="🪵 Logging level."
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        default="standard_welfare",
        help="🗺️ Map name which switches between rulesets.",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        default="games",
        help="📁Folder to save the game to.",
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="💾Save the game to disk (uses W&B run ID).",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--entity",
        dest="entity",
        default=None,
        help="👤Weights & Biases entity name (can be your username). Note you can also use the WANDB_ENTITY env var.",
    )
    parser.add_argument(
        "--project",
        dest="project",
        default=prompts.WANDB_PROJECT,
        help="📝Weights & Biases project name.",
    )
    parser.add_argument(
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        help="⚖️ Disable logging to wandb.",
    )
    parser.add_argument(
        "--max_years",
        dest="max_years",
        type=int,
        default=10,
        help="🗓️ Ends the game after this many years (~3x as many turns).",
    )
    parser.add_argument(
        "--early_stop_max_years",
        dest="early_stop_max_years",
        type=int,
        default=0,
        help="⏱️ Early stop while telling the models the game lasts --max_years long. No effect if 0.",
    )
    parser.add_argument(
        "--max_message_rounds",
        dest="max_message_rounds",
        type=int,
        default=3,
        help="📨Max rounds of messaging per turn.",
    )
    parser.add_argument(
        "--agent_model",
        dest="agent_model",
        default="gpt-4-32k-0613",
        help="🤖Model name to use for the agent. Can be an OpenAI Chat model, 'random', or 'retreats' (contrive a retreat situation).",
    )
    parser.add_argument(
        "--summarizer_model",
        dest="summarizer_model",
        default="gpt-4-32k-0613",
        help="✍️ Model name to use for the message summarizer. Can be an OpenAI Chat model or 'passthrough'.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=0.5,
        help="🌡️ Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        dest="top_p",
        type=float,
        default=0.9,
        help="⚛️ Top-p for nucleus sampling.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
