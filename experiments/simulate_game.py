"""
Language model scaffolding to play Diplomacy.


"""

import argparse
import logging
import os

from diplomacy import Game, Message
from diplomacy.utils.export import to_saved_game_format
import wandb

import constants
import utils
from prompter import Prompter, model_name_to_prompter


def main():
    """Simulate a game of Diplomacy with the given parameters."""
    # Parse args
    args = parse_args()

    # Initialize seed, wandb, game, logger, and prompter
    utils.set_seed(args.seed)

    wandb.init(
        entity=args.entity,
        project=args.project,
        save_code=True,
        config=vars(args),
        mode="disabled" if args.disable_wandb else "online",
        settings=wandb.Settings(code_dir="experiments"),
    )
    assert wandb.run is not None
    game = Game(map_name=args.map_name)
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logger.setLevel(args.log_level)

    prompter: Prompter = model_name_to_prompter(args.model)

    while not game.is_game_done:
        # Cache the list of possible orders for all locations
        possible_orders = game.get_all_possible_orders()

        for power_name, power in game.powers.items():
            # Prompting the model for a response
            prompter_response = prompter.respond(
                power, game, possible_orders, args.max_message_rounds, args.max_years
            )

            # Check how many of the orders were valid
            valid_order_count = 0
            for order in prompter_response.orders:
                word = order.split()
                unit, destination = " ".join(word[:2]), " ".join(word[2:])
                if game._valid_order(power, unit, destination):
                    valid_order_count += 1
            num_orders = len(prompter_response.orders)
            valid_order_ratio = valid_order_count / len(prompter_response.orders)
            logger.info(
                f"{power_name} valid orders: {valid_order_count}/{num_orders} = {valid_order_ratio * 100.0:.2f}%"
            )

            # Set orders
            game.set_orders(power_name, prompter_response.orders)

            # Send messages
            for recipient, message in prompter_response.messages.items():
                game.add_message(
                    Message(
                        sender=power_name,
                        recipient=recipient,
                        message=message,
                        phase=game.get_current_phase(),
                    )
                )

        # Processing the game to move to the next phase
        game.process()

        # Check whether to end the game
        phase = game.get_phase_history()[-1]
        if utils.get_game_year(phase) > args.max_years:
            game._finish([])

        # Log to wandb
        rendered = game.render(incl_abbrev=True)
        log_object = {
            "meta/year_fractional": utils.get_game_fractional_year(phase),
            "board/rendering": wandb.Html(rendered),
        }
        for power in game.powers.values():
            short_name = power.name[:3]
            if game.phase_type == "A":
                log_object[f"score/units/{short_name}"] = len(power.units)
                log_object[f"score/welfare/{short_name}"] = power.welfare_points
            else:
                log_object[f"score/centers/{short_name}"] = len(power.centers)

        wandb.log(log_object)

        # Print some information about the game
        score_string = " ".join(
            [
                f"{power.abbrev}: {len(power.centers)}/{len(power.units)}/{power.welfare_points}"
                for power in game.powers.values()
            ]
        )
        logger.info(f"{phase.name} C/U/W: {score_string}")

    # Exporting the game to disk to visualize (game is appended to file)
    # Alternatively, we can do >> file.write(json.dumps(to_saved_game_format(game)))
    if not args.no_save:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        output_id = "debug" if args.disable_wandb else wandb.run.id
        to_saved_game_format(
            game, output_path=os.path.join(args.output_folder, f"game-{output_id}.json")
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate a game of Diplomacy with the given parameters."
    )
    parser.add_argument("--log_level", dest="log_level", default="INFO")
    parser.add_argument("--map", dest="map_name", default="standard_welfare")
    parser.add_argument("--output_folder", dest="output_folder", default="games")
    parser.add_argument("--no_save", dest="no_save", action="store_true")
    parser.add_argument("--seed", dest="seed", type=int, default=0, help="random seed")
    parser.add_argument("--entity", dest="entity", default="gabrielmukobi")
    parser.add_argument("--project", dest="project", default=constants.WANDB_PROJECT)
    parser.add_argument("--disable_wandb", dest="disable_wandb", action="store_true")
    parser.add_argument("--max_years", dest="max_years", type=int, default=10)
    parser.add_argument(
        "--max_message_rounds", dest="max_message_rounds", type=int, default=1
    )
    parser.add_argument("--model", dest="model", default="gpt-4-0613")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
