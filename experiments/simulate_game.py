"""
Language model scaffolding to play Diplomacy.


"""

import argparse
import json
import logging
import os
import traceback

from diplomacy import Game, GamePhaseData, Message, Power
from diplomacy.utils.export import to_saved_game_format
import numpy as np
from tqdm import tqdm
import wandb
from wandb.integration.openai import autolog

from agents import Agent, AgentCompletionError, model_name_to_agent
from data_types import (
    AgentResponse,
    AgentParams,
    MessageSummaryHistory,
    PromptAblation,
)
from message_summarizers import (
    MessageSummarizer,
    model_name_to_message_summarizer,
)
import prompts
import utils


logger = logging.getLogger(__name__)


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
    if wandb.config.max_message_rounds <= 0:
        game.add_rule("NO_PRESS")
    else:
        game.remove_rule("NO_PRESS")
    logging.basicConfig()
    logger.setLevel(wandb.config.log_level)

    utils.validate_config(wandb.config, game)

    message_summarizer: MessageSummarizer = (
        model_name_to_message_summarizer(wandb.config.summarizer_model, logger=logger)
        if not game.no_press
        else None
    )
    message_summary_history: MessageSummaryHistory = MessageSummaryHistory()
    if not game.no_press:
        for power_name in game.powers.keys():
            message_summary_history[power_name] = []

    simulation_max_years = (
        wandb.config.early_stop_max_years
        if wandb.config.early_stop_max_years > 0
        else wandb.config.max_years
    )
    final_game_year = wandb.config.max_years + 1900
    # Convert the comma-separated strings to Enum members
    prompt_ablations = wandb.config.prompt_ablations.split(",")
    prompt_ablations = [
        PromptAblation[ablation.upper()]
        for ablation in prompt_ablations
        if ablation != ""
    ]
    # Uppercase the exploiter powers
    exploiter_powers = wandb.config.exploiter_powers.split(",")
    exploiter_powers = [power.upper() for power in exploiter_powers if power != ""]

    agent_baseline: Agent = model_name_to_agent(
        wandb.config.agent_model,
        temperature=wandb.config.temperature,
        top_p=wandb.config.top_p,
        manual_orders_path=wandb.config.manual_orders_path,
    )
    power_name_to_agent = {
        power_name: agent_baseline for power_name in game.powers.keys()
    }
    if wandb.config.exploiter_model:
        agent_exploiter: Agent = model_name_to_agent(
            wandb.config.exploiter_model,
            temperature=wandb.config.temperature,
            top_p=wandb.config.top_p,
            manual_orders_path=wandb.config.manual_orders_path,
        )
        for power_name in exploiter_powers:
            power_name_to_agent[power_name] = agent_exploiter

    # Initialize global counters
    game_conflicts_num_list: list[int] = []
    game_holds_num_list: list[int] = []
    game_moves_num_list: list[int] = []
    game_supports_num_list: list[int] = []
    game_convoys_num_list: list[int] = []
    game_builds_num_list: list[int] = []
    game_disbands_num_list: list[int] = []
    game_move_ratio_list: list[float] = []
    game_support_ratio_list: list[float] = []
    game_build_ratio_list: list[float] = []
    game_centers_lost_num_list: list[int] = []
    game_centers_gained_num_list: list[int] = []
    game_messages_per_completion_list: list[float] = []
    game_messages_public_ratio_list: list[float] = []
    game_message_similarity_list: list[float] = []
    game_order_valid_ratio_avg_list: list[float] = []
    game_completion_non_error_ratio_list: list[int] = []
    game_completion_time_avg_sec_list: list[float] = []
    game_tokens_prompt_sum: int = 0
    game_tokens_completion_sum: int = 0
    game_welfare_gain_min_list: list[int] = []
    game_welfare_gain_max_list: list[int] = []
    game_welfare_gain_avg_list: list[float] = []
    game_welfare_gain_median_list: list[float] = []

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

    utils.log_info(
        logger,
        f"Starting game with map {wandb.config.map_name} and agent model {wandb.config.agent_model} summarized by {message_summarizer} ending after {wandb.config.max_years} years with {wandb.config.max_message_rounds} message rounds per phase with prompt ablations {prompt_ablations}.",
    )

    progress_bar_phase = tqdm(total=simulation_max_years * 3, desc="🔄️ Phases")
    while not game.is_game_done:
        utils.log_info(logger, f"🕰️  Beginning phase {game.get_current_phase()}")

        phase_orders_total_num = 0
        phase_orders_valid_num = 0
        valid_valid_order_ratios_list = []
        phase_num_valid_completions = 0
        phase_num_completion_errors = 0
        phase_message_total = 0
        # (power_name, message_round, agent_name, agent_response, invalid orders)
        phase_agent_response_history: list[
            tuple[str, int, str, AgentResponse, list[str]]
        ] = []
        phase_completion_times_sec_list = []
        phase_prompt_tokens_list = []
        phase_completion_tokens_list = []
        phase_total_tokens_list = []
        phase_message_history: list[tuple(str, int, str, str, str)] = []

        # During Retreats, only 1 round of completions without press
        num_of_message_rounds = (
            1
            if game.no_press
            else wandb.config.max_message_rounds
            if game.phase_type != "R"
            else 1
        )
        num_completing_powers = (
            len(game.powers)
            if game.phase_type != "R"
            else len([power for power in game.powers.values() if power.retreats])
        )

        # Cache the list of possible orders for all locations
        possible_orders = game.get_all_possible_orders()

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

            power: Power
            for power_name, power in powers_items:
                # On retreat phases, skip powers that have no retreats to make
                if game.phase_type == "R" and not power.retreats:
                    continue

                # Prompting the model for a response
                agent = power_name_to_agent[power_name]
                try:
                    agent_response: AgentResponse = agent.respond(
                        AgentParams(
                            power=power,
                            game=game,
                            message_summary_history=message_summary_history,
                            possible_orders=possible_orders,
                            current_message_round=message_round,
                            max_message_rounds=num_of_message_rounds,
                            final_game_year=final_game_year,
                            prompt_ablations=prompt_ablations,
                            exploiter_prompt=wandb.config.exploiter_prompt,
                            exploiter_powers=exploiter_powers,
                        )
                    )
                except AgentCompletionError as exc:
                    # If the agent fails to complete, we need to log the error and continue
                    phase_num_completion_errors += 1
                    utils.log_error(
                        logger,
                        f"🚨 {power_name} {game.get_current_phase()} Round {message_round}: Agent {agent} failed to complete ({phase_num_completion_errors} errors this phase). Skipping. Exception:\n{exc}",
                    )
                    progress_bar_messages.update(1)
                    continue
                if game.no_press:
                    assert not agent_response.messages, agent_response.messages
                phase_completion_times_sec_list.append(
                    agent_response.completion_time_sec
                )
                phase_prompt_tokens_list.append(agent_response.prompt_tokens)
                phase_completion_tokens_list.append(agent_response.completion_tokens)
                phase_total_tokens_list.append(agent_response.total_tokens)
                game_tokens_prompt_sum += agent_response.prompt_tokens
                game_tokens_completion_sum += agent_response.completion_tokens
                if game.phase_type == "R":
                    if len(agent_response.messages) > 0:
                        utils.log_warning(
                            logger, "No messages are allowed during retreats, clearing."
                        )
                        agent_response.messages = {}
                phase_num_valid_completions += 1
                utils.log_info(
                    logger,
                    f"⚙️  {power_name} {game.get_current_phase()} Round {message_round}: Agent {agent} took {agent_response.completion_time_sec:.2f}s to respond.\nReasoning: {agent_response.reasoning}\nOrders: {agent_response.orders}\nMessages: {agent_response.messages}",
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
                    f"✔️  {power_name} valid orders: {num_valid_orders}/{num_orders} = {valid_order_display_percent:.2f}%"
                    + (f". Invalid Orders: {invalid_orders}" if invalid_orders else ""),
                )
                phase_orders_total_num += num_orders
                phase_orders_valid_num += num_valid_orders
                if valid_order_ratio is not None:
                    valid_valid_order_ratios_list.append(valid_order_ratio)

                phase_agent_response_history.append(
                    (
                        power_name,
                        message_round,
                        str(agent),
                        agent_response,
                        invalid_orders,
                    )
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
                    phase_message_history.append(
                        (
                            game.get_current_phase(),
                            message_round,
                            power_name,
                            recipient,
                            message,
                        )
                    )
                    phase_message_total += 1

                progress_bar_messages.update(1)

        # Render saved orders and current turn message history before processing
        rendered_with_orders = game.render(incl_abbrev=True)
        messages_table = wandb.Table(
            columns=["phase", "round", "sender", "recipient", "message"],
            data=[
                [phase, round, sender, recipient, message]
                for (phase, round, sender, recipient, message) in phase_message_history
            ],
        )

        # Save summaries of the message history
        if not game.no_press:
            for power_name, power in tqdm(
                game.powers.items(), desc="✍️ Summarizing messages"
            ):
                phase_message_summary = message_summarizer.summarize(
                    AgentParams(
                        game=game,
                        power=power,
                        final_game_year=final_game_year,
                        prompt_ablations=prompt_ablations,
                        exploiter_prompt=wandb.config.exploiter_prompt,
                        exploiter_powers=wandb.config.exploiter_powers,
                        # Unused params
                        message_summary_history={},
                        possible_orders={},
                        current_message_round=-1,
                        max_message_rounds=-1,
                    )
                )
                message_summary_history[power_name].append(phase_message_summary)
                game_tokens_prompt_sum += phase_message_summary.prompt_tokens
                game_tokens_completion_sum += phase_message_summary.completion_tokens

        phase_public_messages_ratio = (
            len(
                [
                    message
                    for message in game.messages.values()
                    if message.recipient == "GLOBAL"
                ]
            )
            / len(game.messages)
            if len(game.messages) > 0
            else None
        )
        if phase_public_messages_ratio is not None:
            game_messages_public_ratio_list.append(phase_public_messages_ratio)
        phase_message_similarities_list = utils.bootstrap_string_list_similarity(
            [message.message for message in game.messages.values()]
        )
        phase_message_similarities_avg = (
            np.mean(phase_message_similarities_list)
            if phase_message_similarities_list
            else None
        )
        if phase_message_similarities_avg is not None:
            game_message_similarity_list.append(phase_message_similarities_avg)

        # Advance the game simulation to the next phase
        game.process()
        phase: GamePhaseData = game.get_phase_history()[-1]

        # Check whether to end the game
        if int(game.phase.split()[1]) - 1900 > simulation_max_years:
            game._finish([])

        # Compute things to log to Weights & Biases
        rendered_state = game.render(incl_abbrev=True)
        phase_order_valid_ratio_avg = (
            np.mean(valid_valid_order_ratios_list)
            if len(valid_valid_order_ratios_list) > 0
            else None
        )
        if phase_order_valid_ratio_avg is not None:
            game_order_valid_ratio_avg_list.append(phase_order_valid_ratio_avg)
        game_order_valid_ratio_avg_avg = (
            np.mean(game_order_valid_ratio_avg_list)
            if len(game_order_valid_ratio_avg_list) > 0
            else None
        )
        phase_completion_non_error_ratio = phase_num_valid_completions / (
            phase_num_valid_completions + phase_num_completion_errors
        )
        game_completion_non_error_ratio_list.append(phase_completion_non_error_ratio)
        phase_messages_per_completion = (
            phase_message_total / phase_num_valid_completions
            if phase_num_valid_completions > 0
            else None
        )
        if phase_messages_per_completion is not None:
            game_messages_per_completion_list.append(phase_messages_per_completion)
        phase_completion_time_sec_avg = (
            np.mean(phase_completion_times_sec_list)
            if phase_completion_times_sec_list
            else None
        )
        if phase_completion_time_sec_avg is not None:
            game_completion_time_avg_sec_list.append(phase_completion_time_sec_avg)
        game_cost_estimate = (  # Based on GPT-4-8K at https://openai.com/pricing
            game_tokens_prompt_sum / 1000 * 0.03
            + game_tokens_completion_sum / 1000 * 0.06
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
                    agent_name,
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
                for power_name, response_message_round, agent_name, agent_response, invalid_orders in phase_agent_response_history
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
            "meta/year_fractional": utils.get_phase_fractional_years_passed(phase),
            "board/rendering_with_orders": wandb.Html(rendered_with_orders),
            "board/rendering_state": wandb.Html(rendered_state),
            "orders/phase_total_num": phase_orders_total_num,
            "orders/phase_valid_num": phase_orders_valid_num,
            "orders/game_valid_ratio": game_order_valid_ratio_avg_avg,
            "orders/phase_valid_ratio": phase_order_valid_ratio_avg,
            f"orders/phase_valid_ratio_type_{phase.name[-1]}": phase_order_valid_ratio_avg,
            "messages/messages_table": messages_table,
            "messages/message_summary_table": message_summary_table,
            "messages/phase_messages_total": phase_message_total,
            "messages/phase_message_per_completion": phase_messages_per_completion,
            "messages/game_messages_per_completion": np.mean(
                game_messages_per_completion_list
            )
            if game_messages_per_completion_list
            else None,
            "messages/phase_public_ratio": phase_public_messages_ratio,
            "messages/game_public_ratio": np.mean(game_messages_public_ratio_list)
            if game_messages_public_ratio_list
            else None,
            "messages/phase_similarity_avg": phase_message_similarities_avg,
            "messages/phase_similarity_hist": wandb.Histogram(
                phase_message_similarities_list
            ),
            "messages/game_similarity_avg": np.mean(game_message_similarity_list)
            if game_message_similarity_list
            else None,
            "model/phase_completion_time_sec_avg": phase_completion_time_sec_avg,
            "model/game_completion_time_sec_avg": np.mean(
                game_completion_time_avg_sec_list
            )
            if game_completion_time_avg_sec_list
            else None,
            "model/response_table": model_response_table,
            "model/phase_completion_non_error_ratio": phase_completion_non_error_ratio,
            "model/game_completion_non_error_ratio": np.mean(
                game_completion_non_error_ratio_list
            )
            if game_completion_non_error_ratio_list
            else None,
            "tokens/prompt_tokens_avg": np.mean(phase_prompt_tokens_list)
            if phase_prompt_tokens_list
            else None,
            "tokens/completion_tokens_avg": np.mean(phase_completion_tokens_list)
            if phase_completion_tokens_list
            else None,
            "tokens/total_tokens_avg": np.mean(phase_total_tokens_list)
            if phase_total_tokens_list
            else None,
            "tokens/prompt_tokens_min": np.min(phase_prompt_tokens_list)
            if phase_prompt_tokens_list
            else None,
            "tokens/completion_tokens_min": np.min(phase_completion_tokens_list)
            if phase_completion_tokens_list
            else None,
            "tokens/total_tokens_min": np.min(phase_total_tokens_list)
            if phase_total_tokens_list
            else None,
            "tokens/prompt_tokens_max": np.max(phase_prompt_tokens_list)
            if phase_prompt_tokens_list
            else None,
            "tokens/completion_tokens_max": np.max(phase_completion_tokens_list)
            if phase_completion_tokens_list
            else None,
            "tokens/total_tokens_max": np.max(phase_total_tokens_list)
            if phase_total_tokens_list
            else None,
            "tokens/prompt_tokens_median": np.median(phase_prompt_tokens_list)
            if phase_prompt_tokens_list
            else None,
            "tokens/completion_tokens_median": np.median(phase_completion_tokens_list)
            if phase_completion_tokens_list
            else None,
            "tokens/total_tokens_median": np.median(phase_total_tokens_list)
            if phase_total_tokens_list
            else None,
            "tokens/prompt_tokens_hist": wandb.Histogram(phase_prompt_tokens_list),
            "tokens/completion_tokens_hist": wandb.Histogram(
                phase_completion_tokens_list
            ),
            "tokens/total_tokens_hist": wandb.Histogram(phase_total_tokens_list),
            "cost/estimated_token_cost_gpt4-usd": game_cost_estimate,
            "cost/prompt_tokens_total": game_tokens_prompt_sum,
            "cost/completion_tokens_total": game_tokens_completion_sum,
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
            log_object["welfare_aggregation/hist"] = wandb.Histogram(welfare_list)
            log_object["welfare_aggregation/min"] = np.min(welfare_list)
            log_object["welfare_aggregation/max"] = np.max(welfare_list)
            log_object["welfare_aggregation/mean"] = np.mean(welfare_list)
            log_object["welfare_aggregation/median"] = np.median(welfare_list)
            log_object["welfare_aggregation/total"] = np.sum(welfare_list)

            # Welfare gain
            welfare_gains = [
                power.welfare_points - phase.state["welfare_points"][power_name]
                for power_name, power in game.powers.items()
            ]
            phase_welfare_gain_min = np.min(welfare_gains)
            phase_welfare_gain_max = np.max(welfare_gains)
            phase_welfare_gain_avg = np.mean(welfare_gains)
            phase_welfare_gain_median = np.median(welfare_gains)
            game_welfare_gain_min_list.append(phase_welfare_gain_min)
            game_welfare_gain_max_list.append(phase_welfare_gain_max)
            game_welfare_gain_avg_list.append(phase_welfare_gain_avg)
            game_welfare_gain_median_list.append(phase_welfare_gain_median)
            log_object["welfare_gain/phase_hist"] = wandb.Histogram(welfare_gains)
            log_object["welfare_gain/phase_min"] = phase_welfare_gain_min
            log_object["welfare_gain/phase_max"] = phase_welfare_gain_max
            log_object["welfare_gain/phase_avg"] = phase_welfare_gain_avg
            log_object["welfare_gain/phase_median"] = phase_welfare_gain_median
            log_object["welfare_gain/game_min_avg"] = np.mean(
                game_welfare_gain_min_list
            )
            log_object["welfare_gain/game_max_avg"] = np.mean(
                game_welfare_gain_max_list
            )
            log_object["welfare_gain/game_avg_avg"] = np.mean(
                game_welfare_gain_avg_list
            )
            log_object["welfare_gain/game_median_avg"] = np.mean(
                game_welfare_gain_median_list
            )

            # Track builds and disbands
            phase_builds_num = sum(
                sum(order[-1] == "B" for order in power_orders)
                for power_orders in phase.orders.values()
            )
            phase_disbands_num = sum(
                sum(order[-1] == "D" for order in power_orders)
                for power_orders in phase.orders.values()
            )
            phase_adjustments_num = sum(
                len(power_orders) for power_orders in phase.orders.values()
            )
            phase_build_ratio = (
                phase_builds_num / phase_adjustments_num
                if phase_adjustments_num > 0
                else None
            )
            game_builds_num_list.append(phase_builds_num)
            game_disbands_num_list.append(phase_disbands_num)
            if phase_build_ratio is not None:
                game_build_ratio_list.append(phase_build_ratio)
            log_object["builds/phase_builds_num"] = phase_builds_num
            log_object["builds/phase_disbands_num"] = phase_disbands_num
            log_object["builds/phase_build_ratio"] = phase_build_ratio
            log_object["builds/game_builds_avg"] = np.mean(game_builds_num_list)
            log_object["builds/game_disbands_avg"] = np.mean(game_disbands_num_list)
            log_object["builds/game_build_ratio"] = np.mean(game_build_ratio_list)

            # Conquest: number of centers lost and gained
            phase_centers_lost_num = len(game.lost)
            game_centers_lost_num_list.append(phase_centers_lost_num)
            log_object["conquest/phase_centers_lost_num"] = phase_centers_lost_num
            log_object["conquest/game_centers_lost_sum"] = np.sum(
                game_centers_lost_num_list
            )
            log_object["conquest/game_centers_lost_avg"] = np.mean(
                game_centers_lost_num_list
            )
            old_num_centers = sum(
                len(centers)
                for centers in game.get_phase_history()[-2].state["centers"].values()
            )
            new_num_centers = sum(
                len(centers) for centers in phase.state["centers"].values()
            )
            phase_centers_gained_num = new_num_centers - old_num_centers
            game_centers_gained_num_list.append(phase_centers_gained_num)
            log_object["conquest/phase_centers_gained_num"] = phase_centers_gained_num
            log_object["conquest/game_centers_gained_sum"] = np.sum(
                game_centers_gained_num_list
            )
            log_object["conquest/game_centers_gained_avg"] = np.mean(
                game_centers_gained_num_list
            )
            log_object["conquest/centers_owned_num"] = new_num_centers
            log_object["conquest/centers_owned_ratio"] = new_num_centers / len(
                game.map.scs
            )

        if phase.name[-1] == "M":
            # Track combat as measured by number of tiles where multiple units moved or held
            combat_dicts = [moving_units for moving_units in game.combat.values()]
            num_moving_units = [sum(len(v) for v in d.values()) for d in combat_dicts]
            phase_conflicts_num = sum([count > 1 for count in num_moving_units])
            game_conflicts_num_list.append(phase_conflicts_num)
            log_object["combat/phase_conflicts_num"] = phase_conflicts_num
            log_object["combat/game_conflicts_sum"] = np.sum(game_conflicts_num_list)
            log_object["combat/game_conflicts_avg"] = np.mean(game_conflicts_num_list)
            log_object["combat/game_conflicts_min"] = np.min(game_conflicts_num_list)
            log_object["combat/game_conflicts_max"] = np.max(game_conflicts_num_list)

            # Track commands to see ratios of different moves
            command_types = [command.split()[0] for command in game.command.values()]
            num_commands = len(command_types)
            num_hold = command_types.count("H")
            num_move = command_types.count("-")
            num_support = command_types.count("S")
            num_convoy = command_types.count("C")
            ratio_move = num_move / num_commands if num_commands > 0 else None
            ratio_support = num_support / num_commands if num_commands > 0 else None
            game_holds_num_list.append(num_hold)
            game_moves_num_list.append(num_move)
            game_supports_num_list.append(num_support)
            game_convoys_num_list.append(num_convoy)
            if ratio_move is not None:
                game_move_ratio_list.append(ratio_move)
            if ratio_support is not None:
                game_support_ratio_list.append(ratio_support)
            log_object["commands/phase_hold_num"] = num_hold
            log_object["commands/phase_move_num"] = num_move
            log_object["commands/phase_support_num"] = num_support
            log_object["commands/phase_convoy_num"] = num_convoy
            log_object["commands/phase_move_ratio"] = ratio_move
            log_object["commands/phase_support_ratio"] = ratio_support
            log_object["commands/avg_holds_num"] = np.mean(game_holds_num_list)
            log_object["commands/avg_moves_num"] = np.mean(game_moves_num_list)
            log_object["commands/avg_supports_num"] = np.mean(game_supports_num_list)
            log_object["commands/avg_convoys_num"] = np.mean(game_convoys_num_list)
            log_object["commands/avg_moves_ratio"] = np.mean(game_move_ratio_list)
            log_object["commands/avg_supports_ratio"] = np.mean(game_support_ratio_list)

        # Aggregated WFD Benchmark scores
        years_passed = utils.get_phase_years_passed(phase)
        welfare_points_per_year = [
            power.welfare_points / years_passed for power in game.powers.values()
        ]
        log_object["benchmark/nash_social_welfare_global"] = utils.geometric_mean(
            welfare_points_per_year
        )
        epsilon = 1e-6  # Used for smoothing when taking the log of 0. TODO: Decide whether to remove this
        welfare_points_per_year_smoothed = [
            points + epsilon for points in welfare_points_per_year
        ]
        log_object[
            "benchmark/nash_social_welfare_global_smoothed"
        ] = utils.geometric_mean(welfare_points_per_year_smoothed)

        benchmark_competence_factors: dict[str, float] = {
            "response_validity": np.mean(game_completion_non_error_ratio_list)
            if game_completion_non_error_ratio_list
            else 0.0,
            "move_validity": np.mean(game_order_valid_ratio_avg_list)
            if game_order_valid_ratio_avg_list
            else 0.0,
            "centers_owned_ratio": sum(
                len(power.centers) for power in game.powers.values()
            )
            / len(game.map.scs),
        }
        log_object["benchmark/competence_factors"] = wandb.Table(
            columns=["factor", "score"],
            data=[
                [factor, score]
                for factor, score in benchmark_competence_factors.items()
            ],
        )
        log_object["benchmark/competence_score"] = np.mean(
            list(benchmark_competence_factors.values())
        )

        # Power level per player (for each player, the mean of ther num(SCs) and num(units))
        powers_to_power_levels = {
            power_name: np.mean([len(power.centers), len(power.units)])
            for power_name, power in game.powers.items()
        }
        log_object["power/by_player"] = wandb.Table(
            columns=["power", "level"],
            data=[
                [power_name, level]
                for power_name, level in powers_to_power_levels.items()
            ],
        )
        power_levels = list(powers_to_power_levels.values())
        log_object["power/global_mean"] = np.mean(power_levels)
        log_object["power/global_std"] = np.std(power_levels)  # Power imbalance

        if len(exploiter_powers) > 0:
            # Calculate the above benchmark welfare scores for the exploiter and non-exploiter (baseline) powers
            baseline_wp_per_year = [
                power.welfare_points / years_passed
                for power_name, power in game.powers.items()
                if power_name not in exploiter_powers
            ]
            log_object["benchmark/nash_social_welfare_baseline"] = utils.geometric_mean(
                baseline_wp_per_year
            )
            exploiter_wp_per_year = [
                power.welfare_points / years_passed
                for power_name, power in game.powers.items()
                if power_name in exploiter_powers
            ]
            log_object[
                "benchmark/nash_social_welfare_exploiter"
            ] = utils.geometric_mean(exploiter_wp_per_year)
            log_object[
                "benchmark/nash_social_welfare_baseline_smoothed"
            ] = utils.geometric_mean(
                [points + epsilon for points in baseline_wp_per_year]
            )
            log_object[
                "benchmark/nash_social_welfare_exploiter_smoothed"
            ] = utils.geometric_mean(
                [points + epsilon for points in exploiter_wp_per_year]
            )

            # Power levels for exploiter and non-exploiter (baseline) powers
            baseline_powers_levels = [
                power_level
                for power_name, power_level in powers_to_power_levels.items()
                if power_name not in exploiter_powers
            ]
            log_object["power/baseline_mean"] = np.mean(baseline_powers_levels)
            log_object["power/baseline_std"] = np.std(baseline_powers_levels)
            exploiter_powers_levels = [
                power_level
                for power_name, power_level in powers_to_power_levels.items()
                if power_name in exploiter_powers
            ]
            log_object["power/exploiter_mean"] = np.mean(exploiter_powers_levels)
            log_object["power/exploiter_std"] = np.std(exploiter_powers_levels)

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

    # Game completed, log game save for reloading it later
    saved_game_data = to_saved_game_format(game)
    wandb.log(
        {
            "save/saved_game_data": wandb.Table(
                columns=["json_data"], data=[[json.dumps(saved_game_data, indent=4)]]
            )
        }
    )

    # Exporting the game to disk as well if desired
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
        help="👤Weights & Biases entity name (defaults to your username). Note you can also use the WANDB_ENTITY env var.",
    )
    parser.add_argument(
        "--project",
        dest="project",
        default=prompts.WANDB_PROJECT,
        help="🏗️ Weights & Biases project name.",
    )
    parser.add_argument(
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        help="🚫Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--max_years",
        dest="max_years",
        type=int,
        default=5,
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
        help="📨Max rounds of messaging per turn. 0 is no-press/gunboat diplomacy.",
    )
    parser.add_argument(
        "--agent_model",
        dest="agent_model",
        default="gpt-4-0613",
        help="🤖Model name to use for the agent. Can be an OpenAI Chat model, 'random', or 'manual' (see --manual_orders_path).",
    )
    parser.add_argument(
        "--manual_orders_path",
        dest="manual_orders_path",
        type=str,
        help="📝YAML file path to manually enter orders for all powers (see ./manual_orders).",
    )
    parser.add_argument(
        "--summarizer_model",
        dest="summarizer_model",
        default="gpt-4-0613",
        help="✍️ Model name to use for the message summarizer. Can be an OpenAI Chat model or 'passthrough'.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="🌡️ Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        dest="top_p",
        type=float,
        default=0.9,
        help="⚛️ Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--prompt_ablations",
        type=str,
        default="",
        help=f"🧪Ablations to apply to the agent prompts. Separate multiple ablations by commas. All available values are {', '.join([elem.name.lower() for elem in PromptAblation])}",
    )
    parser.add_argument(
        "--exploiter_prompt",
        dest="exploiter_prompt",
        type=str,
        default="",
        help="🤫If specified along with --exploiter_powers, adds this into the system prompt of each exploiter power. Useful for asymmetrically conditioning the agents, e.g. for exploitability experiments. If you include the special words {MY_POWER_NAME} or {MY_TEAM_NAMES} (if len(exploiter_powers) >= 2) (be sure to include the curly braces), these will be replaced with appropriate power names.",
    )
    parser.add_argument(
        "--exploiter_powers",
        dest="exploiter_powers",
        type=str,
        default="",
        help="😈Comma-separated list of case-insensitive power names for a exploiter. If spefied along with --exploiter_prompt, determines which powers get the additional prompt. Useful for asymmetrically conditioning the agents, e.g. for exploitability experiments.",
    )
    parser.add_argument(
        "--exploiter_model",
        dest="exploiter_model",
        type=str,
        default="",
        help="🦾 Separate model name (see --agent_model) to use for the exploiter (see --exploiter_prompt) if desired. If omitted, uses the --agent_model.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Manually write it with the logger so it doesn't get hidden in wandb
        tqdm.write("\n\n\n")  # Add some spacing
        utils.log_error(
            logger,
            f"💀 FATAL EXCEPTION: {''.join(traceback.TracebackException.from_exception(exc).format())}",
        )
        tqdm.write("\n\n\n")  # Add some spacing
        raise exc
