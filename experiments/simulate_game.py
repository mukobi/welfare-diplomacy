"""
Language model scaffolding to play Diplomacy.


"""

import argparse
import logging
import os
import random

from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
import wandb

import constants
import utils


def main():
    """Simulate a game of Diplomacy with the given parameters."""
    # Parse args
    args = parse_args()

    # Initialize seed, wandb, game, and logger
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

    # to_saved_game_format(game, output_path=args.output_path)

    # Creating a game
    while not game.is_game_done:
        # Getting the list of possible orders for all locations
        possible_orders = game.get_all_possible_orders()

        # For each power, randomly sampling a valid order
        for power_name, power in game.powers.items():
            power_orders = []
            for loc in game.get_orderable_locations(power_name):
                if possible_orders[loc]:
                    # If this is a disbandable unit in an adjustment phase in welfare,
                    # then randomly choose whether to disband or not
                    if (
                        "ADJUSTMENTS" in str(game.phase)
                        and " D" in possible_orders[loc][0][-2:]
                        and game.welfare
                    ):
                        power_orders.append(
                            random.choice(["WAIVE", possible_orders[loc][0]])
                        )
                    else:
                        power_orders.append(random.choice(possible_orders[loc]))

            game.set_orders(power_name, power_orders)

        # Messages can be sent locally with game.add_message
        # e.g. game.add_message(Message(sender='FRANCE',
        #                               recipient='ENGLAND',
        #                               message='This is a message',
        #                               phase=self.get_current_phase(),
        #                               time_sent=int(time.time())))

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
